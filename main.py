import pickle
import torch
import copy
import pandas as pd
import numpy as np
import reranking
from experiment.metric import *
import json
from common.author import Author
from common.plot import plot_stats
from collections import Counter
from common.reranking import reranking_logic
from team_formation import form_teams_with_skills
from team_formation import convert_author_id_to_attributes
from experiment.experiment import experimental_metrics
vector_data = './data/processed/dblp-toy/teamsvecs.pkl'
splits = dict()
with open(vector_data, 'rb') as infile: vecs = pickle.load(infile)
with open (f'./data/processed/dblp-toy/splits.json','r') as reader:
  splits = json.load(reader)
Y = vecs["member"][splits['test']]
Y_ = torch.load(f'./output/f1.test.pred')
final_reranked_prediction_list = list()
prediction_list = [[float(row[i]) for i in range(len(Y_[0]))] for row in Y_]
prediction_list_copy = copy.deepcopy(prediction_list)
for listItem in prediction_list:
  for index,item in enumerate(listItem):
    if listItem[index] > 0.5: listItem[index] ='P'
    else: listItem[index]= 'NP'
distribution_list = {'NP':0.5,'P':0.5}
result_store = list()
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
  np.append(result_store,[before,after])
  result_store.append([counter,before,after])
  print(f"The NDKL metric for row-{counter} of before and after re-ranking are {before:.3f} and {after:.3f}, respectively.")
ndkl_results = pd.DataFrame(result_store)
final_reranked_prediction_list = np.asarray(final_reranked_prediction_list)
ndkl_results.to_csv('./output/reranked-ndkl.csv',index=None,header=["row number", "NDKL Before", "NDKL After"])
df, df_mean, (fpr, tpr) = calculate_metrics(Y, final_reranked_prediction_list, False)
# auc_score = calculate_metrics(Y, final_reranked_prediction_list, False)
# print(auc_score) #0.6782544378698225
print(df_mean)
df_mean.to_csv(f'./output/test.pred.eval.csv', float_format='%.15f')
