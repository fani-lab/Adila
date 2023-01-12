import json, os, statistics, pandas as pd, pickle
import multiprocessing
from time import time, perf_counter
from functools import partial

import torch
import reranking
from tqdm import tqdm
from scipy.sparse import csr_matrix

from experiment.metric import *


class Reranking:
    @staticmethod
    def get_stats(teamsvecs_members, coefficient: float, output: str) -> tuple:
        """
        Args:
            teamsvecs_members: teamsvecs pickle file
            coefficient: coefficient to calculate a threshold for popularity ( e.g. if 0.5, threshold = 0.5 * average number of teams for a specific member)
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

        labels = ['p' if threshold <= nteam_member else 'np' for nteam_member in col_sums.getA1() ] #rowid maps to columnid in teamvecs['member']
        popular_portion = labels.count('p') / stats['*nmembers']
        stats['popularity_ratio'] = {'p':  popular_portion, 'np': 1 - popular_portion}
        with open(f'{output}stats.pkl', 'wb') as f: pickle.dump(stats, f)
        pd.DataFrame(data=labels, columns=['popularity']).to_csv(f'{output}popularity.csv')
        return stats, labels

    @staticmethod
    def rerank(preds, labels, output, ratios, algorithm: str = 'det_greedy', k_max=5) -> tuple:
        """
        Args:
            preds: loaded predictions from a .pred file
            labels: popularity labels
            output: address of the output directory
            ratios: desired ratio of popular/non-popular items in the output
            algorithm: the chosen algorithm for reranking
            k_max: maximum number of returned team members by reranker
        Returns:
            tuple (list, list)
        """
        idx, probs = list(), list()
        for team in tqdm(preds):
            member_popularity_probs = [(m, labels[m] , float(team[m])) for m in range(len(team))]
            member_popularity_probs.sort(key=lambda x: x[2], reverse=True) #sort based on probs
            #TODO: e.g., please comment the semantics of the output indexes by an example
            #in the output list, we may have an index for a member outside the top k_max list that brought up by the reranker and comes to the top k_max
            start_time = perf_counter()
            reranked_idx = reranking.rerank([label for _, label, _ in member_popularity_probs], ratios, k_max=k_max, algorithm=algorithm)
            finish_time = perf_counter()
            with open(f'../output/fairness.runtime.txt', 'a') as file: file.write(f'{finish_time-start_time} {output} {algorithm} {k_max}\n')
            reranked_probs = [member_popularity_probs[m][2] for m in reranked_idx]
            idx.append(reranked_idx)
            probs.append(reranked_probs)

        pd.DataFrame({'reranked_idx': idx, 'reranked_probs': probs}).to_csv(f'{output}.rerank.{algorithm}.{k_max}.csv')
        return idx, probs

    @staticmethod
    def eval_fairness(preds, labels, reranked_idx, ratios, output, algorithm) -> dict:
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
        dic = {'ndkl.before':[], 'ndkl.after':[]}
        for i, team in enumerate(tqdm(preds)):
            member_popularity_probs = [(m, labels[m], float(team[m])) for m in range(len(team))]
            dic['ndkl.before'].append(reranking.ndkl([label for _, label, _ in member_popularity_probs], ratios))
            dic['ndkl.after'].append(reranking.ndkl([labels[int(m)] for m in reranked_idx[i]], ratios))
        pd.DataFrame(dic).to_csv(f'{output}.faireval.{algorithm}.csv')
        return dic

    @staticmethod
    def reranked_preds(teamsvecs_members, splits, reranked_idx, reranked_probs, output, algorithm) -> csr_matrix:
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
        with open(f'{output}reranked.{algorithm}.preds.pkl', 'wb') as f: pickle.dump(sparse_matrix_reranked, f)
        return sparse_matrix_reranked

    @staticmethod
    def eval_utility(teamsvecs_members, reranked_preds, preds, splits, metrics, output, algorithm, op=('after', 'before')) -> None:
        """
        Args:
            teamsvecs_members: teamsvecs pickle file
            reranked_preds: re-ranked teams
            preds: loaded predictions from a .pred file
            splits: indices of test and train samples
            metrics: desired utility metrics
            output: address of the output directory

        Returns:
            None
        """
        # predictions = torch.load(self.predictions_address)
        y_test = teamsvecs_members[splits['test']]
        if 'before' in op:
            df_, df_mean_, aucroc_, _ = calculate_metrics(y_test, preds, True, metrics) #although we already have this at test.pred.eval.mean.csv
            df_mean_.index.name = 'metric'
            df_mean_.to_csv(f'{output}.utilityeval.before.csv')
        if 'after' in op:
            df, df_mean, aucroc, _ = calculate_metrics(y_test, reranked_preds.toarray(), True, metrics)
            df_mean.index.name = 'metric'
            df_mean.to_csv(f'{output}.utilityeval.{algorithm}.after.csv')

    @staticmethod
    def create_plot(fairness_before: list, fairness_after: list, utility_before, utility_after, legend_material: dict,output, save=False):
        pass
        #color_list = ['red', 'blue', 'green', 'black', 'yellow']
        # custom_scatter_plot = list()
        # custom_scatter_plot.append((self.df_mean_.loc[[utility_metric]], statistics.mean(reranking_results[0])))
        # custom_scatter_plot.append((self.df_mean.loc[[utility_metric]], statistics.mean(reranking_results[1])))
        # before_plot = plt.scatter(custom_scatter_plot[0][1], custom_scatter_plot[0][0], c=color, marker='v')
        # after_plot = plt.scatter(custom_scatter_plot[1][1], custom_scatter_plot[1][0], c=color, marker='o')
        # for fold in zip(fairness_before, utility_before):
        #     plt.scatter(fold[0], fold[1], c=color_list.pop(), marker='v')
        # color_list = ['red', 'blue', 'green', 'black', 'yellow']
        # for fold in zip(fairness_after, utility_after):
        #     plt.scatter(fold[0], fold[1], c=color_list.pop(), marker='o')
        #
        # plt.ylim(ymin=0, ymax=1)
        # plt.xlim(xmin=0, xmax=1)
        # plt.ylabel(legend_material['utility_metric'])
        # plt.xlabel(legend_material['fairness_metric'])
        # if save :
        #     plt.title(f'Utility vs Fairness before and after re-ranking with {legend_material.get("algorithm")}')
        #     #plt.legend((before_plot, after_plot), ('before reranking', 'after reranking'))
        #     plt.savefig(output)
        #     plt.show()

    @staticmethod
    def fairness_average(fairevals: list) -> tuple:

        return statistics.mean([df['ndkl_before'].mean() for df in fairevals]), statistics.mean([df['ndkl_after'].mean() for df in fairevals])

    @staticmethod
    def utility_average(utilityevals: list, metric: str) -> float:

        return statistics.mean([df.loc[df['metric'] == metric, 'mean'].tolist()[0] for df in utilityevals])

    @staticmethod
    def run(fpreds, output, fteamsvecs, fsplits, ratios, fairness_metric={'ndkl'}, utility_metrics={'map_cut_2,5,10'}) -> None:
        """
        Args:
            fpreds: address of the .pred file
            output: address of the output directory
            fteamsvecs: address of teamsvecs file
            fsplits: address of splits.json file
            ratios: desired ratio of popular/non-popular items in the output
            fairness_metric: desired fairness metric
            utility_metrics: desired utility metric

        Returns:
            None
        """
        print('#'*100)
        print(f'Reranking for the baseline {output} ...')
        st = time()
        if not os.path.isdir(output): os.makedirs(output)
        with open(fteamsvecs, 'rb') as f: teamsvecs = pickle.load(f)
        with open(fsplits, 'r') as f: splits = json.load(f)
        preds = torch.load(fpreds)

        try:
            print('Loading popularity labels ...')
            with open(f'{output}stats.pkl', 'rb') as f: stats = pickle.load(f)
            labels = pd.read_csv(f'{output}popularity.csv')['popularity'].to_list()
        except (FileNotFoundError, EOFError):
            print('Loading popularity labels failed, generating popularity labels ...')
            stats, labels = Reranking.get_stats(teamsvecs['member'], coefficient=1, output=output)
        if not ratios: ratios = stats['popularity_ratio']

        new_output = f'{output}/{os.path.split(fpreds)[-1]}'
        algorithm, k_max = 'det_greedy', 5

        try:
            print('Loading re-ranking results ...')
            df = pd.read_csv(f'{new_output}.rerank.{algorithm}.{k_max}.csv', converters={'reranked_idx': eval, 'reranked_probs': eval})
            reranked_idx, probs = df['reranked_idx'].to_list(), df['reranked_probs'].to_list()
        except FileNotFoundError:
            print(f'Loading re-ranking results failed, reranking the predictions based on {algorithm} for top-{k_max} ...')
            reranked_idx, probs = Reranking.rerank(preds, labels, new_output, ratios, algorithm, k_max)

        try:
            print('Loading fairness evaluation results ...')
            fairness_eval = pd.read_csv(f'{new_output}.faireval.{algorithm}.csv')
        except FileNotFoundError:
            print(f'Loading fairness results failed, Evaluating fairness metric {fairness_metric} ...') #for now, it's hardcoded for 'ndkl'
            Reranking.eval_fairness(preds, labels, reranked_idx, ratios, new_output, algorithm)
        try:
            print('Loading re-ranked predictions sparse matrix ...')
            with open(f'{output}reranked.{algorithm}.preds.pkl', 'rb') as f: reranked_preds = pickle.load(f)
        except FileNotFoundError:
            print(' Loading re-ranked predictions sparse matrix failed. Re-ranked predictions sparse matrix reconstruction ...')
            reranked_preds = Reranking.reranked_preds(teamsvecs['member'], splits, reranked_idx, probs, output, algorithm)
        try:
            print('Loading utility metric evaluation results ...')
            utility_before = pd.read_csv(f'{new_output}.utilityeval.before.csv')
        except:
            print(f' Loading utility metric results failed. Evaluating utility metric {utility_metrics} ...')
            Reranking.eval_utility(teamsvecs['member'], reranked_preds, preds, splits, utility_metrics, new_output, algorithm)
        try:
            utility_after =  pd.read_csv(f'{new_output}.utilityeval.{algorithm}.after.csv')
        except:
            print(f' Loading utility metric results failed. Evaluating utility metric {utility_metrics} ...')
            Reranking.eval_utility(teamsvecs['member'], reranked_preds, preds, splits, utility_metrics, new_output, algorithm, op=('after',))

        print(f'Reranking for the baseline {output} completed by {multiprocessing.current_process()}! {time() - st}')
        print('#'*100)

if __name__ == "__main__":
    output = '../output/title.basics.tsv.filtered.mt75.ts3'  # tobe argv
    fsplits = '../data/preprocessed/imdb/title.basics.tsv.filtered.mt75.ts3/splits.json'
    fteamsvecs = '../data/preprocessed/imdb/title.basics.tsv.filtered.mt75.ts3/teamsvecs.pkl'
    files = list()
    for dirpath, dirnames, filenames in os.walk(output):
        files += [os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if file.endswith("pred")]

    files = pd.DataFrame(files, columns=['.', '..', 'domain', 'baseline', 'setting', 'rfile'])
    address_list = list()

    # serial run #todo: by argv
    # for i, row in files.iterrows():
    #     output = f"{row['.']}/{row['..']}/{row['domain']}/{row['baseline']}/{row['setting']}/"
    #     Reranking.run(fpreds=f'{output}/{row["rfile"]}', output=f'{output}/rerank/', fteamsvecs=fteamsvecs, fsplits=fsplits, ratios=None)

    # #parallel run
    ncore = -1 #tobe argv
    with multiprocessing.Pool(multiprocessing.cpu_count() if ncore < 0 else ncore) as executor:
        print(f'Parallel run started ...')
        pairs = []
        for i, row in files.iterrows():
            output = f"{row['.']}/{row['..']}/{row['domain']}/{row['baseline']}/{row['setting']}/"
            pairs.append((f'{output}/{row["rfile"]}', f'{output}/rerank/'))
        executor.starmap(partial(Reranking.run, fteamsvecs=fteamsvecs, fsplits=fsplits, ratios=None), pairs)