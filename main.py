import pickle
import torch
import copy
import pandas as pd
import numpy as np
import reranking
from experiment.metric import *
import json
vector_data = './data/processed/dblp-toy/teamsvecs.pkl'
splits = dict()
with open(vector_data, 'rb') as infile: vecs = pickle.load(infile)
with open (f'./data/processed/dblp-toy/splits.json','r') as reader:
  splits = json.load(reader)
Y = vecs["member"][splits['test']]
Y_ = torch.load(f'./data/processed/dblp-toy/fnn/basic/f0.test.pred')
final_reranked_prediction_list = list()
prediction_list = [[float(row[i]) for i in range(len(Y_[0]))] for row in Y_]
prediction_list_copy = copy.deepcopy(prediction_list)
for listItem in prediction_list:
  for index,item in enumerate(listItem):
    if listItem[index] > 0.5: listItem[index] ='P'
    else: listItem[index]= 'NP'
distribution_list = {'NP':0.5,'P':0.5}
result_store = list()
ndkl_before = list()
ndkl_after = list()
for counter,row_list,deep_copy in zip(range(len(prediction_list)),prediction_list,prediction_list_copy):
  ranking_indices= reranking.rerank(
    row_list,
    distribution_list,
    max_na=None,
    k_max=None,
    algorithm="det_greedy",
  )
  item_attribute_reranked = [row_list[i] for i in ranking_indices]
  reranked_to_numbers = [deep_copy[i] for i in ranking_indices]
  final_reranked_prediction_list.append(reranked_to_numbers)
  before = reranking.ndkl(row_list, distribution_list)
  after = reranking.ndkl(item_attribute_reranked, distribution_list)
  torch.save(final_reranked_prediction_list, f'./output/fnn/basic/f0.test.rrk.pred', pickle_protocol=4)

  ndkl_before.append([counter,before])
  ndkl_after.append([counter, after])
  # print(f"The NDKL metric for row-{counter} of before and after re-ranking are {before:.3f} and {after:.3f}, respectively.")
ndkl_before_result = pd.DataFrame(ndkl_before).to_csv('./output/fnn/basic/f0.test.pred.fair.csv',index=None,header=["row number", "NDKL Before",])
ndkl_after_result = pd.DataFrame(ndkl_after).to_csv('./output/fnn/basic/f0.test.pred.rrk.fair.csv',index=None,header=["row number", "NDKL After"])
final_reranked_prediction_list = np.asarray(final_reranked_prediction_list)
df, df_mean, (fpr, tpr) = calculate_metrics(Y, final_reranked_prediction_list, False)
print(df_mean)
df_mean.to_csv(f'./output/fnn/basic/f0.test.rrk.pred.eval.csv', float_format='%.15f')