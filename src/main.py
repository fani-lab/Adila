import json, os, statistics, pandas as pd, pickle, multiprocessing, argparse, warnings, numpy as np, random
from time import time, perf_counter
from functools import partial
from tqdm import tqdm
from random import randrange
from scipy.sparse import csr_matrix
import torch

import fairsearchcore as fsc
import FairRankTune as frt
from fairsearchcore.models import FairScoreDoc
import reranking

from cmn.metric import *
#from util.fair_greedy import fairness_greedy
from util.visualization import area_under_curve

warnings.simplefilter(action='ignore', category=FutureWarning)

class Reranking:

    @staticmethod
    def gender_process(fgender: str, output: str):
        try: ig = pd.read_csv(f'{output}/labels.csv')
        except FileNotFoundError:
            ig = pd.read_csv(fgender)
            random.seed(42)  # To make the results reproducible
            # For now only false is given since in uspt the gender bias is extreme 93% male in the random case
            ig.fillna(random.choice([False]), inplace=True)
            ig = ig.rename(columns={'Unnamed: 0': 'memberidx'})
            ig.sort_values(by='memberidx', inplace=True)
            ig.to_csv(f'{output}/labels.csv', index=False)
        index_female = ig.loc[ig['gender'] == False, 'memberidx'].tolist()
        index_male = ig.loc[ig['gender'] == True, 'memberidx'].tolist()
        gender_ratio = len(index_female) / (len(index_female) + len(index_male))
        return ig, gender_ratio

    @staticmethod
    def get_stats(teamsvecs, fgender, coefficient: float, output: str, fairness_notion: str = 'dp', att='popularity', popularity_thresholding: str ='avg') -> tuple:
        """
        Args:
            teamsvecs_members: teamsvecs pickle file
            coefficient: coefficient to calculate a threshold for popularity (e.g. if 0.5, threshold = 0.5 * average number of teams for a specific member)
            output: address of the output directory
            fairness_notion: dp: demographic parity, eo: equality of opportunity
            popularity_thresholding: argument to select the method we label popular vs nonpopular experts ('avg' or 'auc')
        Returns:
             tuple (dict, list)

        """
        teamids, skillvecs, teamsvecs_members = teamsvecs['id'], teamsvecs['skill'], teamsvecs['member']
        stats = {}
        stats['*nmembers'] = teamsvecs_members.shape[1]
        col_sums = teamsvecs_members.sum(axis=0)

        stats['nteams_candidate-idx'] = {k: v for k, v in enumerate(sorted(col_sums.A1.astype(int), reverse=True))}
        stats['*avg_nteams_member'] = col_sums.mean()

        x, y = zip(*enumerate(sorted(col_sums.A1.astype(int), reverse=True)))
        stats['*auc_nteams_member'] =  area_under_curve(x, y, 'expert-idx', 'nteams', show_plot=False)

        threshold = coefficient * stats[f'*{popularity_thresholding}_nteams_member']

        if att == 'popularity':
            #TODO: many nonpopular but few popular. So, we only keep popular labels. Though the protected group is the nonpopulars.
            #TODO: this should be the same for all baselines, so read the file from ./output/{domain}
            labels = [True if threshold <= nteam_member else False for nteam_member in col_sums.getA1() ] #rowid maps to columnid in teamvecs['member']
            stats['np_ratio'] = labels.count(False) / stats['*nmembers']
            with open(f'{output}/stats.pkl', 'wb') as f: pickle.dump(stats, f)
            pd.DataFrame(data=labels, columns=['popularity']).to_csv(f'{output}/labels.csv', index_label='memberidx')
            #TODO: we read it again!
            sensitive_att = pd.read_csv(f'{output}/labels.csv')

        elif att == 'gender':
            #TODO: many males but few females. So, we keep the females. Also, females are protected groups.
            #TODO: this should be the same for all baselines, so read the file from ./output/{domain}
            sensitive_att, stats['np_ratio'] = Reranking.gender_process(fgender, output)
            with open(f'{output}/stats.pkl', 'wb') as f: pickle.dump(stats, f)
            labels = sensitive_att['gender'].tolist()

        if fairness_notion == 'eo':
            skill_member = skillvecs.transpose() @ teamsvecs_members
            ratios = list()
            print("Generating ratios ... ")
            for i in tqdm(range(skillvecs.shape[0])):
                skill_indexes = skillvecs[i].nonzero()[1].tolist()
                members = [skill_member[idx].nonzero()[1] for idx in skill_indexes]
                intersect = set(members[0]).union(*members)
                # to avoid empty set
                if len(intersect) == 0: intersect = [randrange(0, teamsvecs_members.shape[1]) for i in range(5)]

                #TODO: need to be changed if we only keep the labels of minority group
                member_dict = dict(zip(sensitive_att['memberidx'], sensitive_att[att]))
                # Retrieve 'gender' values for members in 'intersect'
                labels_ = [member_dict.get(member, None) for member in intersect]
                ratios.append(labels_.count(False) / len(intersect))

            with open(f'{output}/ratios.pkl', 'wb') as file: pickle.dump(ratios, file)
            return stats, labels, ratios

        else: return stats, labels, None # None is to unify the number of returned arguments by the function to avoid complications in run function

    @staticmethod
    def rerank(preds, labels, output, ratios, algorithm: str = 'det_greedy', k_max: int = None, fairness_notion: str = 'dp', alpha: float = 0.05, att: str = 'popularity', popularity_thresholding: str ='avg' ) -> tuple:
        """
        Args:
            preds: loaded predictions from a .pred file
            labels: popularity labels
            output: address of the output directory
            ratios: desired ratio of popular/non-popular items in the output
            algorithm: ranker algorithm of choice among {'det_greedy', 'det_cons', 'det_relaxed', 'fa-ir'}
            k_max: maximum number of returned team members by reranker
            cutoff: to resize the list of experts before giving it to the re-ranker
            alpha: significance value for fa*ir algorithm
        Returns:
            tuple (list, list)
        """
        start_time = perf_counter()
        r = ratios
        temp = r[False]
        if temp < 0.2: temp = 0.2
        elif temp > 0.98: temp = 0.98
        fair = fsc.Fair(k_max, temp, alpha)
        idx, probs, protected = list(), list(), list()
        for i, team in enumerate(tqdm(preds)):
            member_probs = [(m, labels[m], float(team[m])) for m in range(len(team))]
            member_probs.sort(key=lambda x: x[2], reverse=True)
            # The usage of not operator is because we mapped popular as True and non-popular as False.
            # Non-popular is our protected group and vice versa. So we need to use not in FairScoreDocs
            if fairness_notion == 'eo':
                r = {True: 1 - ratios[i], False: ratios[i]}
                temp = r[False]
                if temp < 0.2: temp = 0.2
                elif temp > 0.98: temp = 0.98
                fair = fsc.Fair(k_max, temp, alpha) #fair.p = r; fair._cache = {}
            if algorithm == 'fa-ir':
                fair_doc = [FairScoreDoc(m[0], m[2], not m[1]) for m in member_probs]
                # fair = fsc.Fair(k_max, ratios[i], alpha)
                if fair.is_fair(fair_doc[:k_max]): reranked = fair_doc[:k_max] #no change
                else: reranked = fair.re_rank(fair_doc)[:k_max]
                idx.append([x.id for x in reranked])
                probs.append([x.score for x in reranked])

            elif algorithm in ['det_greedy', 'det_relaxed', 'det_cons', 'det_const_sort']:
                reranked_idx = reranking.rerank([label for _, label, _ in member_probs], r, k_max=k_max, algorithm=algorithm)
                reranked_probs = [member_probs[m][2] for m in reranked_idx]
                idx.append(reranked_idx)
                probs.append(reranked_probs)
            elif algorithm == 'fair_greedy':
                #TODO refactor and parameterize this algorithm
                bias_dict = dict([(member_probs.index(m), {'att': m[1], 'prob': m[2], 'idx': m[0]}) for m in member_probs[:500]])
                method = 'move_down'
                reranked_idx = fairness_greedy(bias_dict, r, 'att', method)[:k_max]
                reranked_probs = [bias_dict[idx]['prob'] for idx in reranked_idx[:k_max]]
                idx.append(reranked_idx)
                probs.append(reranked_probs)
            else: raise ValueError('chosen reranking algorithm is not valid')
        pd.DataFrame({'reranked_idx': idx, 'reranked_probs': probs}).to_csv(f'{output}.{algorithm}.{popularity_thresholding+"."  if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "") + "." if algorithm=="fa-ir" else ""}{k_max}.rerank.csv', index=False)
        return idx, probs, (perf_counter() - start_time)

    @staticmethod
    def calculate_prob(atr: bool, team: list) -> float: return team.count(atr) / len(team)

    @staticmethod
    def eval_fairness(preds, labels, reranked_idx, ratios, output, algorithm, k_max, alpha, fairness_notion: str = 'dp', metrics: set = {'skew', 'ndkl'}, att: str = 'popularity', popularity_thresholding: str ='avg' ):
        """
        Args:
            preds: loaded predictions from a .pred file
            labels: popularity labels
            reranked_idx: indices of re-ranked teams with a pre-defined cut-off
            ratios: desired ratio of popular/non-popular items in the output
            output: address of the output directory
        Returns:
            dict: ndkl metric before and after re-ranking
        """

        # because the mapping between popular/nonpopular and protected/nonprotected is reversed
        # TODO also check if we need more specific file names ( with fairness criteria for example)
        # use argument instead of this line
        # if algorithm == 'fa-ir':
        #     labels = [not value for value in labels]
        dic_before, dic_after = dict(), dict()
        for metric in metrics:
            dic_before[metric], dic_after[metric] = list(), list()
            if metric in ['skew', 'exp', 'expu']: dic_before[metric], dic_after[metric] = {'protected': [], 'nonprotected': []}, {'protected': [], 'nonprotected': []}
            for i, team in enumerate(tqdm(preds)):
                # defining the threshold for the times we have or don't have cutoff
                threshold = len(preds) if k_max is None else k_max

                if fairness_notion == 'eo': r = {True: 1 - ratios[i], False: ratios[i]}
                else: r = ratios
                member_probs = [(m, labels[m], float(team[m])) for m in range(len(team))]
                member_probs.sort(key=lambda x: x[2], reverse=True)
                #IMPORTANT: the ratios keys should match the labels!
                if 'ndkl' == metric:
                    dic_before[metric].append(reranking.ndkl([label for _, label, _ in member_probs[:threshold]], r))
                    dic_after[metric].append(reranking.ndkl([labels[int(m)] for m in reranked_idx[i]], r))

                if 'skew' == metric:
                    l_before = [label for _, label, _ in member_probs[: threshold]]
                    l_after = [labels[int(m)] for m in reranked_idx[i]]
                    dic_before['skew']['protected'].append(reranking.skew(Reranking.calculate_prob(False, l_before), r[False]))
                    dic_before['skew']['nonprotected'].append(reranking.skew(Reranking.calculate_prob(True, l_before), r[True]))
                    dic_after['skew']['protected'].append(reranking.skew(Reranking.calculate_prob(False, l_after), r[False]))
                    dic_after['skew']['nonprotected'].append(reranking.skew(Reranking.calculate_prob(True, l_after), r[True]))

                if metric in ['exp', 'expu']:
                    #TODO Needs Refactor
                    if metric == 'exp':
                        exp_before, per_group_exp_before = frt.Metrics.EXP(pd.DataFrame(data=[j[0] for j in member_probs[:k_max]]), dict([(j[0], j[1]) for j in member_probs[:k_max]]), 'MinMaxRatio')
                    elif metric == 'expu':
                        exp_before, per_group_exp_before = frt.Metrics.EXPU(pd.DataFrame(data=[j[0] for j in member_probs[:k_max]]), dict([(j[0], j[1]) for j in member_probs[:k_max]]), pd.DataFrame(data=[j[2] for j in member_probs[:k_max]]),'MinMaxRatio')
                    else: raise ValueError('Chosen Metric Is not Valid')

                    try: dic_before[metric]['protected'].append(per_group_exp_before[False])
                    except KeyError:dic_before[metric]['protected'].append(0)
                    try: dic_before[metric]['nonprotected'].append(per_group_exp_before[True])
                    except KeyError: dic_before[metric]['nonprotected'].append(0)
                    dic_before[metric][metric] = exp_before

                    if metric == 'exp':
                        exp_after, per_group_exp_after = frt.Metrics.EXP(pd.DataFrame(data=reranked_idx[i][:k_max]), dict([(j, labels[j]) for j in reranked_idx[i][:k_max]]), 'MinMaxRatio')
                        # dic_after[metric]['protected'].append(per_group_exp_after[False]), dic_after[metric]['nonprotected'].append(per_group_exp_after[True])
                        # dic_after[metric][metric] = exp_after
                    elif metric == 'expu':
                        exp_after, per_group_exp_after = frt.Metrics.EXPU(pd.DataFrame(data=reranked_idx[i][:k_max]), dict([(j, labels[j]) for j in reranked_idx[i][:k_max]]), pd.DataFrame(data=[j[2] for i in reranked_idx[i][:k_max] for j in member_probs if j[0] == i]), 'MinMaxRatio')
                        # dic_after[metric]['protected'].append(per_group_exp_after[False]), dic_after[metric]['nonprotected'].append(per_group_exp_after[True])
                        # dic_after[metric][metric] = exp_after
                    else:raise ValueError('Chosen Metric Is not Valid')
                    try: dic_after[metric]['protected'].append(per_group_exp_after[False])
                    except KeyError: dic_after[metric]['protected'].append(0)
                    try:  dic_after[metric]['nonprotected'].append(per_group_exp_after[True])
                    except KeyError:  dic_after[metric]['nonprotected'].append(0)
                    dic_after[metric][metric] = exp_after

            df_before = pd.DataFrame(dic_before[metric]).mean(axis=0).to_frame('mean.before')
            df_after = pd.DataFrame(dic_after[metric]).mean(axis=0).to_frame('mean.after')
            df = pd.concat([df_before, df_after], axis=1)
            df.to_csv(f'{output}.{algorithm}.{popularity_thresholding+"."  if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "")+"." if algorithm=="fa-ir" else ""}{k_max}.{metric}.faireval.csv', index_label='metric')

    @staticmethod
    def reranked_preds(teamsvecs_members, splits, reranked_idx, reranked_probs, output, algorithm, k_max, alpha, att: str = 'popularity', popularity_thresholding: str ='avg') -> csr_matrix:
        """
        Args:
            teamsvecs_members: teamsvecs pickle file
            splits: indices of test and train samples
            reranked_idx: indices of re-ranked teams with a pre-defined cut-off
            reranked_probs: original probability of re-ranked items
            output: address of the output directory

        Returns:
            csr_matrix
        """
        y_test = teamsvecs_members[splits['test']]
        rows, cols, value = list(), list(), list()
        for i, reranked_team in enumerate(tqdm(reranked_idx)):
            for j, reranked_member in enumerate(reranked_team):
                rows.append(i)
                cols.append(reranked_member)
                value.append(reranked_probs[i][j])
        sparse_matrix_reranked = csr_matrix((value, (rows, cols)), shape=y_test.shape)
        with open(f'{output}.{algorithm}.{popularity_thresholding+"." if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "")+"." if algorithm=="fa-ir" else ""}{k_max}.rerank.pred', 'wb') as f: pickle.dump(sparse_matrix_reranked, f)
        return sparse_matrix_reranked

    @staticmethod
    def eval_utility(teamsvecs_members, reranked_preds, fpred, preds, splits, metrics, output, algorithm, k_max, alpha,  att: str = 'popularity', popularity_thresholding: str ='avg' ) -> None:
        """
        Args:
            teamsvecs_members: teamsvecs pickle file
            reranked_preds: re-ranked teams
            fpred: .pred filename (to see if .pred.eval.mean.csv exists)
            preds: loaded predictions from a .pred file
            splits: indices of test and train samples
            metrics: desired utility metrics
            output: address of the output directory

        Returns:
            None
        """
        y_test = teamsvecs_members[splits['test']]
        try: df_mean_before = pd.read_csv(f'{fpred}.eval.mean.csv', names=['mean'], header=0)#we should already have it at f*.test.pred.eval.mean.csv
        except FileNotFoundError:
            _, df_mean_before, _, _ = calculate_metrics(y_test, preds, False, metrics)
            df_mean_before.to_csv(f'{fpred}.eval.mean.csv', columns=['mean'])
        df_mean_before.rename(columns={'mean': 'mean.before'}, inplace=True)
        _, df_mean_after, _, _ = calculate_metrics(y_test, reranked_preds.toarray(), False, metrics)
        df_mean_after.rename(columns={'mean': 'mean.after'}, inplace=True)
        pd.concat([df_mean_before, df_mean_after], axis=1).to_csv(f'{output}.{algorithm}.{popularity_thresholding+"." if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "")+"." if algorithm=="fa-ir" else ""}{k_max}.utileval.csv', index_label='metric')

    @staticmethod
    def run(fpred, output, fteamsvecs, fsplits, fgender,
            fairness_notion='eo', att='popularity', algorithm='det_cons',
            k_max=None, alpha: float = 0.1, np_ratio=None, popularity_thresholding='avg',
            fairness_metrics={'ndkl', 'skew'}, utility_metrics={'ndcg_cut_20,50,100'}

    ) -> None:
        """
        Args:
            fpred: address of the .pred file
            fteamsvecs: address of teamsvecs file
            fsplits: address of splits.json file
            fgender: address of the gender label files
            fairness_notion: chosen notion of fairness 'eo' or 'dp'
            att: chosen sensitive attribute ( 'popularity', 'gender')
            algorithm: ranker algorithm of choice among {'det_greedy', 'det_cons', 'det_relaxed', 'fa-ir'}
            np_ratio: desired ratio of non-popular experts in the output
            k_max: chosen cutoff ( should be an integer less than the size of the team)
            fairness_metrics: desired fairness metric
            utility_metrics: desired utility metric
            output: address of the output directory

        Returns:
            None
        """
        print('#'*100)
        print(f'Reranking for {att} using {algorithm} for the top-{k_max} on the baseline {fpred} ...')
        st = time()
        output += f'{att}'
        if not os.path.isdir(output): os.makedirs(output)
        with open(fteamsvecs, 'rb') as f: teamsvecs = pickle.load(f)
        with open(fsplits, 'r') as f: splits = json.load(f)
        preds = torch.load(fpred)

        try:
            print(f'Loading stats, labels {", ratios" if fairness_notion == "eo" else ""}  ...')
            with open(f'{output}/stats.pkl', 'rb') as f: stats = pickle.load(f)
            labels = pd.read_csv(f'{output}/labels.csv')[att].to_list()
            if fairness_notion == 'eo':
                with open(f'{output}/ratios.pkl', 'rb') as f: ratios = pickle.load(f)
        except (FileNotFoundError, EOFError):
            print(f'Loading failed! Generating files {output} ...')
            stats, labels, ratios = Reranking.get_stats(teamsvecs, fgender, coefficient=1, output=output, fairness_notion=fairness_notion, att=att, popularity_thresholding=popularity_thresholding)

        output += f'/{fairness_notion}'
        #creating a static ratio in case fairness_notion is 'dp'
        if fairness_notion == 'dp':
            if not np_ratio: ratios = {True: 1 - stats['np_ratio'], False: stats['np_ratio']}
            else: ratios = {True: 1 - np_ratio, False: np_ratio}
            assert np.sum(list(ratios.values())) == 1.0
        else: pass

        new_output = f'{output}/{os.path.split(fpred)[-1]}'
        try:
            print('Loading reranking results ...')
            df = pd.read_csv(f'{new_output}.{algorithm}.{popularity_thresholding+"." if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "")+"." if algorithm=="fa-ir" else ""}{k_max}.rerank.csv', converters={'reranked_idx': eval, 'reranked_probs': eval})
            reranked_idx, probs = df['reranked_idx'].to_list(), df['reranked_probs'].to_list()
        except FileNotFoundError:
            print(f'Loading re-ranking results failed! Reranking the predictions based on {att} with {algorithm} for top-{k_max} ...')
            reranked_idx, probs, elapsed_time = Reranking.rerank(preds, labels, new_output, ratios, algorithm, k_max, fairness_notion, alpha, att, popularity_thresholding)
            #not sure os handles file locking for append during parallel run ...
            # with open(f'{new_output}.rerank.time', 'a') as file: file.write(f'{elapsed_time} {new_output} {algorithm} {k_max}\n')
            with open(f'{output}/rerank.time', 'a') as file: file.write(f'{elapsed_time} {new_output} {algorithm} {k_max}\n')
        try:
            with open(f'{output}.{algorithm}.{popularity_thresholding+"." if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "")+"." if algorithm=="fa-ir" else ""}{k_max}.rerank.pred', 'rb') as f: reranked_preds = pickle.load(f)
        except FileNotFoundError: reranked_preds = Reranking.reranked_preds(teamsvecs['member'], splits, reranked_idx, probs, new_output, algorithm, k_max, alpha, att, popularity_thresholding)

        try:
            print('Loading fairness evaluation results before and after reranking ...')
            for metric in fairness_metrics:
                fairness_eval = pd.read_csv(f'{new_output}.{algorithm}.{popularity_thresholding+"." if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "")+"." if algorithm=="fa-ir" else ""}{k_max}.{metric}.faireval.csv')
        except FileNotFoundError:
            print(f'Loading fairness results failed! Evaluating fairness metric {fairness_metrics} ...')
            Reranking.eval_fairness(preds, labels, reranked_idx, ratios, new_output, algorithm, k_max, alpha, fairness_notion, fairness_metrics, att, popularity_thresholding)

        try:
            print('Loading utility metric evaluation results before and after reranking ...')
            utility_before = pd.read_csv(f'{new_output}.{algorithm}.{popularity_thresholding+"." if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "")+"." if algorithm=="fa-ir" else ""}{k_max}.utileval.csv')
        except:
            print(f' Loading utility metric results failed! Evaluating utility metric {utility_metrics} ...')
            Reranking.eval_utility(teamsvecs['member'], reranked_preds, fpred, preds, splits, utility_metrics, new_output, algorithm, k_max, alpha, att, popularity_thresholding)

        print(f'Pipeline for the baseline {fpred} completed by {multiprocessing.current_process()}! {time() - st}')
        print('#'*100)

    @staticmethod
    def addargs(parser):
        dataset = parser.add_argument_group('dataset')
        dataset.add_argument('-fteamsvecs', '--fteamsvecs', type=str, required=True, help='teamsvecs (pickle of a dictionary for three lil_matrix for teamids (1×n), skills(n×s), and members(n×m)) file; required; Eg. -fteamvecs ./data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl')
        dataset.add_argument('-fsplits', '--fsplits', type=str, required=True, help='splits.json for test rowids in teamsvecs and pred file; required; Eg. -fsplits output/toy.dblp.v12.json/splits.json')
        dataset.add_argument('-fpred', '--fpred', type=str, required=True, help='.pred file (torch ndarray (test×m)) or root directory of *.pred files; required; Eg. -fpred ./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/f0.test.pred)')
        dataset.add_argument('-fgender', '--fgender', type=str, required=False, help='')#TODO
        dataset.add_argument('-output', '--output', type=str, required=True, help='output directory')

        fairness = parser.add_argument_group('fairness')
        fairness.add_argument('-fairness_notion', '--fairness_notion', type=str, default='eo', help='eo: equality of opportunity, dp: demographic parity')
        fairness.add_argument('-att', '--att', type=str, required=True, help='protected attribute: popularity or gender')
        fairness.add_argument('-algorithm', '--algorithm', type=str, required=True, help='reranking algorithm from {fa-ir, det_greedy, det_cons, det_relaxed}; required; Eg. det_cons')

"""
A running example of arguments
# single *.pred file
python -u main.py 
-fteamsvecs ../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl
-fsplit ../output/toy.dblp.v12.json/splits.json
-fpred ../output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/f0.test.pred 
-fairness_notion eo
-att popularity
-algorithm det_cons
-output ../output/toy.dblp.v12.json/

# root folder containing many *.pred files.
python -u main.py 
-fteamsvecs ../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl
-fsplit ../output/toy.dblp.v12.json/splits.json
-fpred ../output/toy.dblp.v12.json/
-fairness_notion eo
-att popularity
-algorithm det_cons
-output ../output/toy.dblp.v12.json/
"""
def test_toy_all():
    import params
    for alg in ['det_greedy', 'det_relaxed', 'det_cons', 'det_const_sort', 'fa-ir']:
        for notion in ['eo', 'dp']:
            for att in ['popularity', 'gender']:
                for th in ['avg', 'auc']:
                    params.settings['fair']['popularity_thresholding'] = th
                    params.settings['fair']['k_max'] = 10
                    Reranking.run(fpred='../output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/f0.test.pred',
                                  output='../output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/',
                                  fteamsvecs='../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl',
                                  fsplits='../output/toy.dblp.v12.json/splits.json',
                                  fgender='../data/preprocessed/dblp/toy.dblp.v12.json/gender.csv',
                                  fairness_notion=notion,
                                  att=att,
                                  algorithm=alg,
                                  k_max=params.settings['fair']['k_max'],
                                  alpha=params.settings['fair']['alpha'],
                                  np_ratio=params.settings['fair']['np_ratio'],
                                  popularity_thresholding=params.settings['fair']['popularity_thresholding'],
                                  fairness_metrics=params.settings['fair']['metrics'],
                                  utility_metrics=params.settings['utility_metrics'])

if __name__ == "__main__":
    import params
    # test_toy_all()
    # exit(0)

    parser = argparse.ArgumentParser(description='Fair Team Formation')
    Reranking.addargs(parser)
    args = parser.parse_args()

    if os.path.isfile(args.fpred):
        Reranking.run(fpred=args.fpred,
                      output=args.output,
                      fteamsvecs=args.fteamsvecs,
                      fsplits=args.fsplits,
                      fgender=args.fgender,
                      fairness_notion=args.fairness_notion,
                      att=args.att,
                      algorithm=args.algorithm,
                      k_max=params.settings['fair']['k_max'],
                      alpha=params.settings['fair']['alpha'],
                      np_ratio=params.settings['fair']['np_ratio'],
                      popularity_thresholding=params.settings['fair']['popularity_thresholding'],
                      fairness_metrics=params.settings['fair']['metrics'],
                      utility_metrics=params.settings['utility_metrics'])
        exit(0)

    if os.path.isdir(args.fpred):
        # given a root folder, we can crawl the folder to find *.pred files and run the pipeline for all
        files = list()
        for dirpath, dirnames, filenames in os.walk(args.fpred): files += [os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if file.endswith("pred") and 'rerank' not in file]

        files = pd.DataFrame(files, columns=['.', '..', 'domain', 'baseline', 'setting', 'rfile'])
        address_list = list()

        pairs = []
        for i, row in files.iterrows():
            output = f"{row['.']}/{row['..']}/{row['domain']}/{row['baseline']}/{row['setting']}/"
            pairs.append((f'{output}{row["rfile"]}', f'{output}rerank/'))

        if params.settings['parallel']:
            print(f'Parallel run started ...')
            with multiprocessing.Pool(multiprocessing.cpu_count() if params.settings['core'] < 0 else params.settings['core']) as executor:
                executor.starmap(partial(Reranking.run,
                                         fsplits=args.fsplits,
                                         fairness_notion=args.fairness_notion,
                                         att=args.att,
                                         fgender=args.fgender,
                                         algorithm=args.algorithm,
                                         k_max=params.settings['fair']['k_max'],
                                         alpha=params.settings['fair']['alpha'],
                                         np_ratio=params.settings['fair']['np_ratio'],
                                         popularity_thresholding=params.settings['fair']['popularity_thresholding'],
                                         fairness_metrics=params.settings['fair']['metrics'],
                                         fteamsvecs=args.fteamsvecs,
                                         utility_metrics=params.settings['utility_metrics']), pairs)
        else:
            for fpred, output in pairs: Reranking.run(fpred=fpred,
                                                      output=output,
                                                      fteamsvecs=args.fteamsvecs,
                                                      fsplits=args.fsplits,
                                                      fgender=args.fgender,
                                                      fairness_notion=args.fairness_notion,
                                                      att=args.att,
                                                      algorithm=args.algorithm,
                                                      k_max=params.settings['fair']['k_max'],
                                                      alpha=params.settings['fair']['alpha'],
                                                      np_ratio=params.settings['fair']['np_ratio'],
                                                      popularity_thresholding=params.settings['fair']['popularity_thresholding'],
                                                      fairness_metrics=params.settings['fair']['metrics'],
                                                      utility_metrics=params.settings['utility_metrics'])
