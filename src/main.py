import copy
import json
import torch
import pickle
import reranking
import pandas as pd


class Reranking:

    def __init__(self,  vectorized_dataset_address: str, splits_address: str, predictions_address: str):
        self.predictions_address = predictions_address
        self.vectorized_dataset_address = vectorized_dataset_address
        self.splits_address = splits_address
        self.splits = None
        self.stats = dict()
        self.teamsvecs = None
        self.predictions = None

    def dataset_stats(self, coefficient: float):
        """
        Args:
            coefficient: the coefficient to define popularity

        Returns:
            stats: dict
        """
        with open(self.vectorized_dataset_address, 'rb') as infile:
            self.teamsvecs = pickle.load(infile)
        teamids, skillvecs, membervecs = self.teamsvecs['id'], self.teamsvecs['skill'], self.teamsvecs['member']

        self.stats['*nteams'] = teamids.shape[0]
        self.stats['*nmembers'] = membervecs.shape[1]
        self.stats['*nskills'] = skillvecs.shape[1]
        # row_sums = membervecs.sum(axis=1)
        col_sums = membervecs.sum(axis=0)

        self.stats['nteams_candidate-idx'] = {k: v for k, v in enumerate(sorted(col_sums.A1.astype(int), reverse=True))}
        self.stats['*avg_nteams_member'] = col_sums.mean()
        threshold = coefficient * self.stats['*avg_nteams_member']
        self.stats['popularity'] = list()

        for index, appearance in zip(teamids, col_sums.getA1()):
            if threshold <= appearance:
                self.stats['popularity'].append('P')
            else:
                self.stats['popularity'].append('NP')

        popular_portion = self.stats['popularity'].count('P') / self.stats['*nmembers']
        self.stats['distributions'] = {'P':  popular_portion, 'NP': 1 - popular_portion}
        return self.stats

    def perform_reranking(self, algorithm: str = 'det_greedy', k_max=5, distribution_list=None):
        """

        Args:
            algorithm: the chosen algorithm for reranking
            k_max: maximum number of returned team members by reranker
            distribution_list: the desired distribution of attributes

        Returns:
            tuple
        """

        with open(self.splits_address, 'r') as reader:
            self.splits = json.load(reader)
        # testset_indexes = self.teamsvecs["member"][self.splits['test']]
        self.predictions = torch.load(self.predictions_address)
        final_reranked_prediction_list = list()
        metric_before = list()
        metric_after = list()
        plt_metric_before = list()
        plt_metric_after = list()

        prediction_list = [[float(row[i]) for i in range(len(self.predictions[0]))] for row in self.predictions]
        prediction_list_copy = copy.deepcopy(prediction_list)
        for counter, row_list, deep_copy in zip(range(len(prediction_list)), prediction_list, prediction_list_copy):
            index_pred = list(enumerate(row_list))
            index_pred.sort(key=lambda x: x[1], reverse=True)
            popularity = list()
            for member in index_pred:
                popularity.append(self.stats['popularity'][member[0]])

            ranking_indices = reranking.rerank(popularity, distribution_list, k_max=k_max, algorithm=algorithm)
            reranked_to_numbers = [deep_copy[i] for i in ranking_indices]
            final_reranked_prediction_list.append(reranked_to_numbers)

            before = reranking.ndkl(popularity, distribution_list)

            after = reranking.ndkl([self.stats['popularity'][i] for i in ranking_indices], distribution_list)

            plt_metric_before.append(before)
            plt_metric_after.append(after)
            metric_before.append([counter, before])
            metric_after.append([counter, after])

        metric_before_result = pd.DataFrame(metric_before)
        metric_after_result = pd.DataFrame(metric_after)

        return plt_metric_before, plt_metric_after, metric_before_result, metric_after_result, final_reranked_prediction_list


# Sample code for running and debugging
reranking_object = Reranking(vectorized_dataset_address='../processed/dblp-toy/teamsvecs.pkl',
                             splits_address=f'../processed/dblp-toy/splits.json',
                             predictions_address=f'../output/toy.dblp.v12.json/fnn/t31.s11.m13.l[100].lr0.1.b4096.e20/f0.test.pred')

reranking_object.dataset_stats(1)
reranking_object.perform_reranking(distribution_list=reranking_object.stats['distributions'])
