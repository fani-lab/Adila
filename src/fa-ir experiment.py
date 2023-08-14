import pickle
import torch
import os, json
from tqdm import tqdm
import fairsearchcore as fsc
from fairsearchcore.models import FairScoreDoc

from main import Reranking


output = '../output/imdb/bnn'
fteamsvecs = '../output/imdb/teamsvecs.pkl'
fsplits = '../output/imdb/splits.json'
fpred = '../output/imdb/bnn/t32059.s23.m2011.l[100].lr0.1.b4096.e20.nns3.nsuniform/f0.test.pred'

print('#' * 100)
if not os.path.isdir(output): os.makedirs(output)
with open(fteamsvecs, 'rb') as f: teamsvecs = pickle.load(f)
with open(fsplits, 'r') as f: splits = json.load(f)
preds = torch.load(fpred)


stats, labels, ratios = Reranking.get_stats(teamsvecs, coefficient=1, output=output, eq_op=False)
fair_docs = list()
r = {True: 1 - stats['np_ratio'], False: stats['np_ratio']}

# Converting our teams into lists of FairScoreDoc
for team in tqdm(preds):
    member_popularity_probs = [(m, labels[m], float(team[m])) for m in range(len(team))]
    member_popularity_probs.sort(key=lambda x: x[2], reverse=True)
    fair_docs.append([FairScoreDoc(m[0], m[2], m[1]) for m in member_popularity_probs])

k = 50 # number of topK elements returned (value should be between 10 and 400)
p = stats['np_ratio'] # proportion of protected candidates in the topK elements (value should be between 0.02 and 0.98)
alpha = 0.1 # significance level (value should be between 0.01 and 0.15)


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

for team in fair_docs:
    print(fair.is_fair(team[:k]))
