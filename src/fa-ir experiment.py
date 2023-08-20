import pickle
import torch
import os, json
import reranking
from tqdm import tqdm
import fairsearchcore as fsc
from fairsearchcore.models import FairScoreDoc

from main import Reranking


output = '../output/imdb/bnn_emb/fa-ir'
fteamsvecs = '../output/imdb/teamsvecs.pkl'
fsplits = '../output/imdb/splits.json'
fpred = '../output/imdb/bnn/t32059.s100.m2011.l[100].lr0.1.b4096.e20.nns3.nsunigram_b/f0.test.pred'

print('#' * 100)
if not os.path.isdir(output): os.makedirs(output)
with open(fteamsvecs, 'rb') as f: teamsvecs = pickle.load(f)
with open(fsplits, 'r') as f: splits = json.load(f)
preds = torch.load(fpred)


stats, labels, ratios = Reranking.get_stats(teamsvecs, coefficient=1, output=output, eq_op=False)
fair_docs = list()
r = {False: 1 - stats['np_ratio'], True: stats['np_ratio']}

dic_before = {'ndkl':[]}; dic_after={'ndkl':[]}

# Converting our teams into lists of FairScoreDoc
for team in tqdm(preds):
    member_popularity_probs = [(m, labels[m], float(team[m])) for m in range(len(team))]
    member_popularity_probs.sort(key=lambda x: x[2], reverse=True)
    dic_before['ndkl'].append(reranking.ndkl([label for _, label, _ in member_popularity_probs], r))
    fair_docs.append([FairScoreDoc(m[0], m[2], not m[1]) for m in member_popularity_probs])

k = 100 # number of topK elements returned (value should be between 10 and 400)
p = stats['np_ratio'] # proportion of protected candidates in the topK elements (value should be between 0.02 and 0.98)
alpha = 0.05 # significance level (value should be between 0.01 and 0.15)


# create the Fair object
fair = fsc.Fair(k, p, alpha)
# create an mtable using alpha unadjusted
mtable = fair.create_unadjusted_mtable()
# analytically calculate the fail probability
analytical = fair.compute_fail_probability(mtable)
# create an mtable using alpha adjusted
mtable = fair.create_adjusted_mtable()
# again, analytically calculate the fail probability
analytical_ = fair.compute_fail_probability(mtable)

fair_teams = list()
labels_ = [not value for value in labels]
# Check to see if a team needs reranking to become fair or not.
print('Analyzing fairness and reranking if necessary...')
for i, team in enumerate(fair_docs):

    if fair.is_fair(team[:k]):
        fair_teams.append(team[:k])
        #dic_after['ndkl'].append(reranking.ndkl([label for _, label, _ in member_popularity_probs], r))
    else:
        print(fair.is_fair(team[:k]))
        reranked = fair.re_rank(team)
        fair_teams.append(reranked[:k])
        #print(dic_before['ndkl'][i])
print('Done !')


print('Creating inputs for sparse matrix creation...')
idx, probs, protected = list(), list(), list()
for fair_team in fair_teams:
    idx.append([x.id for x in fair_team])
    probs.append([x.score for x in fair_team])
    protected.append([x.is_protected for x in fair_team])
print('Done !')

sparse = Reranking.reranked_preds(teamsvecs['member'], splits, idx, probs, output, f'fa-ir.{alpha}', k)
new_output = f'{output}/{os.path.split(fpred)[-1]}'
Reranking.eval_fairness(preds, labels_, idx, r, new_output, f'fa-ir.{alpha}', k)
Reranking.eval_utility(teamsvecs['member'], sparse, fpred, preds, splits, {'map_cut_2,5,10', 'ndcg_cut_2,5,10'}, new_output, f'fa-ir.{alpha}', k)
