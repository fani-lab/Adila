import pickle, json, copy
import torch
import pandas as pd
import numpy as np
import reranking
from experiment.metric import *
from common.metric_plot import *
baseline = '../output/toy.dblp.v12.json/fnn/t31.s11.m13.l[100].lr0.1.b4096.e20/'
vector_data = '../processed/dblp-toy/teamsvecs.pkl'
splits = dict()
with open(vector_data, 'rb') as infile: vecs = pickle.load(infile)
with open(f'../processed/dblp-toy/splits.json', 'r') as reader: splits = json.load(reader)

Y = vecs["member"][splits['test']]
Y_ = torch.load(f'{baseline}f0.test.pred')

final_reranked_prediction_list = list()
prediction_list = [[float(row[i]) for i in range(len(Y_[0]))] for row in Y_]
prediction_list_copy = copy.deepcopy(prediction_list)

for listItem in prediction_list:
    for index, item in enumerate(listItem):
        listItem[index] = 'P' if listItem[index] > 0.5 else 'NP'

distribution_list = {'NP': 0.5, 'P': 0.5}
result_store = list()
ndkl_before = list()
ndkl_after = list()
plt_ndkl_before = list()
plt_ndkl_after = list()

for counter, row_list, deep_copy in zip(range(len(prediction_list)), prediction_list, prediction_list_copy):
    ranking_indices = reranking.rerank(row_list, distribution_list, max_na=None, k_max=None, algorithm="det_greedy")
    item_attribute_reranked = [row_list[i] for i in ranking_indices]
    reranked_to_numbers = [deep_copy[i] for i in ranking_indices]
    final_reranked_prediction_list.append(reranked_to_numbers)
    before = reranking.ndkl(row_list, distribution_list)
    after = reranking.ndkl(item_attribute_reranked, distribution_list)
    torch.save(final_reranked_prediction_list, f'{baseline}f0.test.pred.rrk', pickle_protocol=4)
    plt_ndkl_before.append(before)
    plt_ndkl_after.append(after)
    ndkl_before.append([counter, before])
    ndkl_after.append([counter, after])
    # print(f"The NDKL metric for row-{counter} of before and after re-ranking are {before:.3f} and {after:.3f}, respectively.")

ndkl_before_result = pd.DataFrame(ndkl_before).to_csv(f'{baseline}f0.test.pred.fair.csv', index=None, header=["row number", "NDKL Before", ])
ndkl_after_result = pd.DataFrame(ndkl_after).to_csv(f'{baseline}f0.test.pred.rrk.fair.csv', index=None, header=["row number", "NDKL After"])

final_reranked_prediction_list = np.asarray(final_reranked_prediction_list)
df, df_mean, aucroc, (fpr, tpr) = calculate_metrics(Y, final_reranked_prediction_list, False)
df_mean.to_csv(f'{baseline}f0.test.pred.rrk.eval.csv', float_format='%.15f')

# plt.plot(fpr, tpr, label=f'micro-average on reranked set', linestyle=':', linewidth=4)
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.savefig(f'{baseline}roc-curve.png')
df_mean_before = pd.read_csv(f'{baseline}f0.test.pred.eval.mean.csv')
ndcg_metric_before = np.array(df_mean_before.iloc[6:9,1])
ndcg_metric_after = np.array(df_mean.iloc[6:9,0])
# metric_plot(plt_ndkl_before,plt_ndkl_after, baseline,'ndkl')
metric_plot(ndcg_metric_before,ndcg_metric_after,baseline,'ndcg')