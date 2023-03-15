import json, os, statistics, pandas as pd, pickle, multiprocessing, argparse, numpy as np
from time import time, perf_counter
from functools import partial
from tqdm import tqdm
from scipy.sparse import csr_matrix
import torch

import reranking
from cmn.metric import *

class Reranking:
    @staticmethod
    def get_stats(teamsvecs_members, coefficient: float, output: str) -> tuple:
        """
        Args:
            teamsvecs_members: teamsvecs pickle file
            coefficient: coefficient to calculate a threshold for popularity (e.g. if 0.5, threshold = 0.5 * average number of teams for a specific member)
            output: address of the output directory
        Returns:
             tuple (dict, list)

        """
        stats = {}
        stats['*nmembers'] = teamsvecs_members.shape[1]
        col_sums = teamsvecs_members.sum(axis=0)

        stats['nteams_candidate-idx'] = {k: v for k, v in enumerate(sorted(col_sums.A1.astype(int), reverse=True))}
        stats['*avg_nteams_member'] = col_sums.mean()
        threshold = coefficient * stats['*avg_nteams_member']

        labels = [True if threshold <= nteam_member else False for nteam_member in col_sums.getA1() ] #rowid maps to columnid in teamvecs['member']
        stats['np_ratio'] = labels.count(False) / stats['*nmembers']
        with open(f'{output}stats.pkl', 'wb') as f: pickle.dump(stats, f)
        pd.DataFrame(data=labels, columns=['popularity']).to_csv(f'{output}popularity.csv', index_label='memberidx')
        return stats, labels

    @staticmethod
    def rerank(preds, labels, output, ratios, algorithm: str = 'det_greedy', k_max=5, cutoff=None) -> tuple:
        """
        Args:
            preds: loaded predictions from a .pred file
            labels: popularity labels
            output: address of the output directory
            ratios: desired ratio of popular/non-popular items in the output
            algorithm: the chosen algorithm for reranking in {'det_greedy', 'det_cons', 'det_relaxed'}
            k_max: maximum number of returned team members by reranker
            cutoff: to resize the list of experts before giving it to the re-ranker
        Returns:
            tuple (list, list)
        """
        idx, probs = list(), list()
        start_time = perf_counter()
        for team in tqdm(preds):
            member_popularity_probs = [(m, labels[m], float(team[m])) for m in range(len(team))]
            member_popularity_probs.sort(key=lambda x: x[2], reverse=True) #sort based on probs

            if cutoff is None:
                pass
            else:
                member_popularity_probs = member_popularity_probs[:cutoff]
            #TODO: e.g., please comment the semantics of the output indexes by an example
            #in the output list, we may have an index for a member outside the top k_max list that brought up by the reranker and comes to the top k_max
            reranked_idx = reranking.rerank([label for _, label, _ in member_popularity_probs], ratios, k_max=k_max, algorithm=algorithm)
            reranked_probs = [member_popularity_probs[m][2] for m in reranked_idx]
            idx.append(reranked_idx)
            probs.append(reranked_probs)
        finish_time = perf_counter()
        pd.DataFrame({'reranked_idx': idx, 'reranked_probs': probs}).to_csv(f'{output}.{algorithm}.{k_max}.rerank.csv', index=False)
        return idx, probs, (finish_time - start_time)

    @staticmethod
    def eval_fairness(preds, labels, reranked_idx, ratios, output, algorithm, k_max) -> dict:
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
        dic_before = {'ndkl':[]}; dic_after={'ndkl':[]}
        for i, team in enumerate(tqdm(preds)):
            member_popularity_probs = [(m, labels[m], float(team[m])) for m in range(len(team))]
            member_popularity_probs.sort(key=lambda x: x[2], reverse=True)
            dic_before['ndkl'].append(reranking.ndkl([label for _, label, _ in member_popularity_probs], ratios))
            dic_after['ndkl'].append(reranking.ndkl([labels[int(m)] for m in reranked_idx[i]], ratios))

        df_before = pd.DataFrame(dic_before).mean(axis=0).to_frame('mean.before')
        df_after = pd.DataFrame(dic_after).mean(axis=0).to_frame('mean.after')
        df = pd.concat([df_before, df_after], axis=1)
        df.to_csv(f'{output}.{algorithm}.{k_max}.faireval.csv', index_label='metric')
        return df

    @staticmethod
    def reranked_preds(teamsvecs_members, splits, reranked_idx, reranked_probs, output, algorithm, k_max) -> csr_matrix:
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
                #value.append(y_test[i, reranked_member])
                value.append(reranked_probs[i][j])
        sparse_matrix_reranked = csr_matrix((value, (rows, cols)), shape=y_test.shape)
        with open(f'{output}.{algorithm}.{k_max}.rerank.pred', 'wb') as f: pickle.dump(sparse_matrix_reranked, f)
        return sparse_matrix_reranked

    @staticmethod
    def eval_utility(teamsvecs_members, reranked_preds, fpred, preds, splits, metrics, output, algorithm, k_max) -> None:
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
        # predictions = torch.load(self.predictions_address)
        y_test = teamsvecs_members[splits['test']]
        try:
            df_mean_before = pd.read_csv(f'{fpred}.eval.mean.csv', names=['mean'], header=0)#we should already have it at f*.test.pred.eval.mean.csv
        except FileNotFoundError:
            _, df_mean_before, _, _ = calculate_metrics(y_test, preds, False, metrics)
            df_mean_before.to_csv(f'{fpred}.eval.mean.csv', columns=['mean'])
        df_mean_before.rename(columns={'mean': 'mean.before'}, inplace=True)
        _, df_mean_after, _, _ = calculate_metrics(y_test, reranked_preds.toarray(), False, metrics)
        df_mean_after.rename(columns={'mean': 'mean.after'}, inplace=True)
        pd.concat([df_mean_before, df_mean_after], axis=1).to_csv(f'{output}.{algorithm}.{k_max}.utileval.csv', index_label='metric')

    @staticmethod
    def fairness_average(fairevals: list) -> tuple:
        return statistics.mean([df['ndkl.before'].mean() for df in fairevals]), statistics.mean([df['ndkl.after'].mean() for df in fairevals])

    @staticmethod
    def utility_average(utilityevals: list, metric: str) -> float:
        return statistics.mean([df.loc[df['metric'] == metric, 'mean'].tolist()[0] for df in utilityevals])

    @staticmethod
    def run(fpred, output, fteamsvecs, fsplits, np_ratio, algorithm='det_cons', k_max=None, fairness_metrics={'ndkl'}, utility_metrics={'map_cut_2,5,10'}) -> None:
        """
        Args:
            fpred: address of the .pred file
            output: address of the output directory
            fteamsvecs: address of teamsvecs file
            fsplits: address of splits.json file
            ratio: desired ratio of non-popular experts in the output
            algorithm: ranker algorithm of choice among {'det_greedy', 'det_cons', 'det_relaxed'}
            k_max:
            fairness_metrics: desired fairness metric
            utility_metrics: desired utility metric

        Returns:
            None
        """
        print('#'*100)
        print(f'Reranking for the baseline {fpred} ...')
        st = time()
        if not os.path.isdir(output): os.makedirs(output)
        with open(fteamsvecs, 'rb') as f: teamsvecs = pickle.load(f)
        with open(fsplits, 'r') as f: splits = json.load(f)
        preds = torch.load(fpred)

        try:
            print('Loading popularity labels ...')
            with open(f'{output}stats.pkl', 'rb') as f: stats = pickle.load(f)
            labels = pd.read_csv(f'{output}popularity.csv')['popularity'].to_list()
        except (FileNotFoundError, EOFError):
            print(f'Loading popularity labels failed! Generating popularity labels at {output}stats.pkl ...')
            stats, labels = Reranking.get_stats(teamsvecs['member'], coefficient=1, output=output)
        if not np_ratio: ratios = {'p': 1 - stats['np_ratio'], 'np': stats['np_ratio']}
        else: ratios = {'p': 1 - np_ratio, 'np': np_ratio}
        assert np.sum(list(ratios.values())) == 1.0

        new_output = f'{output}/{os.path.split(fpred)[-1]}'

        try:
            print('Loading reranking results ...')
            df = pd.read_csv(f'{new_output}.{algorithm}.{k_max}.rerank.csv', converters={'reranked_idx': eval, 'reranked_probs': eval})
            reranked_idx, probs = df['reranked_idx'].to_list(), df['reranked_probs'].to_list()
        except FileNotFoundError:
            print(f'Loading re-ranking results failed! Reranking the predictions based on {algorithm} for top-{k_max} ...')
            reranked_idx, probs, elapsed_time = Reranking.rerank(preds, labels, new_output, ratios, algorithm, k_max)
            #not sure os handles file locking for append during parallel run ...
            # with open(f'{new_output}.rerank.time', 'a') as file: file.write(f'{elapsed_time} {new_output} {algorithm} {k_max}\n')
            with open(f'{output}/rerank.time', 'a') as file: file.write(f'{elapsed_time} {new_output} {algorithm} {k_max}\n')
        try:
            with open(f'{new_output}.{algorithm}.{k_max}.rerank.pred', 'rb') as f: reranked_preds = pickle.load(f)
        except FileNotFoundError: reranked_preds = Reranking.reranked_preds(teamsvecs['member'], splits, reranked_idx, probs, new_output, algorithm, k_max)

        try:
            print('Loading fairness evaluation results before and after reranking ...')
            fairness_eval = pd.read_csv(f'{new_output}.{algorithm}.{k_max}.faireval.csv')
        except FileNotFoundError:
            print(f'Loading fairness results failed! Evaluating fairness metric {fairness_metrics} ...') #for now, it's hardcoded for 'ndkl'
            Reranking.eval_fairness(preds, labels, reranked_idx, ratios, new_output, algorithm, k_max)

        try:
            print('Loading utility metric evaluation results before and after reranking ...')
            utility_before = pd.read_csv(f'{new_output}.{algorithm}.{k_max}.utileval.csv')
        except:
            print(f' Loading utility metric results failed! Evaluating utility metric {utility_metrics} ...')
            Reranking.eval_utility(teamsvecs['member'], reranked_preds, fpred, preds, splits, utility_metrics, new_output, algorithm, k_max=k_max)

        print(f'Pipeline for the baseline {fpred} completed by {multiprocessing.current_process()}! {time() - st}')
        print('#'*100)

    @staticmethod
    def addargs(parser):
        dataset = parser.add_argument_group('dataset')
        dataset.add_argument('-fteamsvecs', '--fteamsvecs', type=str, required=True, help='teamsvecs (pickle of a dictionary for three lil_matrix for teamids (1×n), skills(n×s), and members(n×m)) file; required; Eg. -fteamvecs ./data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl')
        dataset.add_argument('-fsplits', '--fsplits', type=str, required=True, help='splits.json for test rowids in teamsvecs and pred file; required; Eg. -fsplits output/toy.dblp.v12.json/splits.json')
        dataset.add_argument('-fpred', '--fpred', type=str, required=True, help='.pred file (torch ndarray (test×m)) or root directory of *.pred files; required; Eg. -fpred ./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/f0.test.pred)')
        dataset.add_argument('-output', '--output', type=str, required=True, help='output directory')

        fairness = parser.add_argument_group('fairness')
        fairness.add_argument('-np_ratio', '--np_ratio', type=float, default=None, required=False, help='desired ratio of non-popular experts after reranking; if None, based on distribution in dataset; default: None; Eg. 0.5')
        dataset.add_argument('-fairness_metrics', '--fairness_metrics', nargs='+', type=set, default={'ndkl'}, required=False, help='list of fairness metrics; default: ndkl')
        dataset.add_argument('-reranker', '--reranker', type=str, required=True, help='reranking algorithm from {det_greedy, det_cons, det_relaxed}; required; Eg. det_cons')
        dataset.add_argument('-k_max', '--k_max', type=int, default=10, required=False, help='cutoff for the reranking algorithms; default: 10')
        dataset.add_argument('-utility_metrics', '--utility_metrics', nargs='+', type=set, default={'map_cut_2,5,10'}, required=False, help='list of utility metric in the form of pytrec_eval; default: map_cut_2,5,10')

        mode = parser.add_argument_group('mode')
        mode.add_argument('-mode', type=int, default=1, choices=[0, 1], help='0 for sequential run and 1 for parallel; default: 1')
        mode.add_argument('-core', type=int, default=-1, help='number of cores to dedicate to parallel run, -1 means all available cores; default: -1')

"""
A running example of arguments
# single *.pred file
python -u main.py 
-fteamsvecs ../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl
-fsplit ../output/toy.dblp.v12.json/splits.json
-fpred ../output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/f0.test.pred 
-reranker det_cons
-output ../output/toy.dblp.v12.json/

# root folder containing many *.pred files.
python -u main.py 
-fteamsvecs ../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl
-fsplit ../output/toy.dblp.v12.json/splits.json
-fpred ../output/toy.dblp.v12.json/
-reranker det_cons
-output ../output/toy.dblp.v12.json/
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fair Team Formation')
    Reranking.addargs(parser)
    args = parser.parse_args()

    if os.path.isfile(args.fpred):
        Reranking.run(fpred=args.fpred,
                      output=args.output,
                      fteamsvecs=args.fteamsvecs,
                      fsplits=args.fsplits,
                      np_ratio=args.np_ratio,
                      algorithm=args.reranker,
                      k_max=args.k_max,
                      fairness_metrics=args.fairness_metrics,
                      utility_metrics=args.utility_metrics)
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
            pairs.append((f'{output}/{row["rfile"]}', f'{output}/rerank/'))

        if args.mode == 0: # sequential run
            for fpred, output in pairs: Reranking.run(fpred=fpred,
                                                      output=output,
                                                      fteamsvecs=args.fteamsvecs,
                                                      fsplits=args.fsplits,
                                                      np_ratio=args.np_ratio,
                                                      algorithm=args.reranker,
                                                      k_max=args.k_max,
                                                      fairness_metrics=args.fairness_metrics,
                                                      utility_metrics=args.utility_metrics)
        elif args.mode == 1: # parallel run
            print(f'Parallel run started ...')
            with multiprocessing.Pool(multiprocessing.cpu_count() if args.core < 0 else args.core) as executor:
                executor.starmap(partial(Reranking.run,
                                         fteamsvecs=args.fteamsvecs,
                                         fsplits=args.fsplits,
                                         np_ratio=args.np_ratio,
                                         algorithm=args.reranker,
                                         k_max=args.k_max,
                                         fairness_metrics=args.fairness_metrics,
                                         utility_metrics=args.utility_metrics
                                         ), pairs)
