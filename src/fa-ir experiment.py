import math
import pickle

import fairsearchcore as fsc
import torch

from main import Reranking
from tqdm import tqdm
from fairsearchcore.models import FairScoreDoc
k = 13 # number of topK elements returned (value should be between 10 and 400)
p = 0.25 # proportion of protected candidates in the topK elements (value should be between 0.02 and 0.98)
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
analytical = fair.compute_fail_probability(mtable)
with open('../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl', 'rb') as f: teamsvecs = pickle.load(f)
stats, labels = Reranking.get_stats(teamsvecs_members=teamsvecs['member'],
                                    coefficient=1, output='../output/toy.dblp.v12.json/FA-IR')
preds = torch.load('../output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/f0.test.pred')

for team in tqdm(preds):
    new_list = list()
    member_popularity_probs = [(m, labels[m], float(team[m])) for m in range(len(team))]
    member_popularity_probs.sort(key=lambda x: x[2], reverse=True)  # sort based on probs
    for member in member_popularity_probs:
        if member[2] == 'p':
            protected = False
        else:
            protected = True
        new_list.append(FairScoreDoc(member[0],math.log(member[2]), protected))
    print(fair.is_fair(new_list))
print('yo')