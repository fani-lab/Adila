# ``Adila``<sup>*</sup>: Fairness-Aware Team Formation
<sup>*[ عادلة, feminine Arabic given name, meaning just and fair](https://en.wikipedia.org/wiki/Adila_(name))<sup>

`Team Formation` aims to automate forming teams of experts who can successfully solve difficult tasks. While state-of-the-art neural team formation methods are able to efficiently analyze massive collections of experts to form effective collaborative teams, they largely ignore the fairness in recommended teams of experts. Fairness breeds innovation and increases teams' success by enabling a stronger sense of community, reducing conflict, and stimulating more creative thinking. In `Adila`, we study the application of `fairness-aware` team formation algorithms to mitigate the potential popularity bias in the neural team formation models. Our experiments show that, first, neural team formation models are biased toward `popular` and `male` experts. Second, although deterministic re-ranking algorithms mitigate `popularity` XOR `gender` bias substantially, they severely hurt the efficacy of teams. On the other hand, probabilistic greedy re-ranking algorithms mitigate `popularity` bias significantly and maintain utility. Finally, due to extreme bias in the dataset in terms of `gender`, probabilistic greedy re-ranking algorithms also fail to achieve fair and efficient teams. 

> We have studied the application of state-of-the-art [`deterministic greedy re-ranking methods [Geyik et al. KDD'19]`](https://dl.acm.org/doi/10.1145/3292500.3330691) in addition to [`probabilistic greedy re-ranking methods [Zehlike et al. IP&M'22]`](https://dl.acm.org/doi/abs/10.1016/j.ipm.2021.102707)to mitigate `populairty bias` and `gender bias` based on `equality of opportunity` and `demographic parity` notions of fairness for state-of-the-art neural team formation methods from [`OpeNTF`](https://github.com/fani-lab/opeNTF/). Our experiments show that:
> - Neural team formation models are biased toward popular experts;
> - Although deterministic re-ranking algorithms mitigate bias substantially, they severely hurt the efficacy of teams.
> - Probabilistic greedy re-ranking methods are able to mitigate bias while maintaining the utility of the teams as well. 

> Currently, we are investigating:
> - Other fairness factors like demographic attributes, including age, race, and gender; 
> - Developing machine learning-based models using Learning-to-Rank (L2R) techniques to mitigate bias as opposed to deterministic greedy algorithms.

- [1. Setup](#1-setup)
- [2. Quickstart](#2-quickstart)
- [3. Pipeline](#3-pipeline)
  * [3.1. Labeling](#31-labeling)
  * [3.2. Gender Distribution](#32-gender)
  * [3.3. Reranking](#33-reranking)
  * [3.4. Evaluations](#34-evaluations)
- [4. Result](#4-result)
- [5. Acknowledgement](#5-acknowledgement)
- [6. License](#6-license)
- [7. Citation](#7-citation)

## 1. Setup
`Adila` needs ``Python=3.8`` and others packages listed in [``requirements.txt``](requirements.txt):

By ``pip``, clone the codebase and install the required packages:
```sh
git clone https://github.com/Fani-Lab/Adila
cd Adila
pip install -r requirements.txt
```

By [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone https://github.com/Fani-Lab/Adila
cd Adila
conda env create -f environment.yml
conda activate adila
```

## 2. Quickstart
To run `Adila`, you can use [`./src/main.py`](./src/main.py):

```bash
cd src
python -u main.py \
  -fteamsvecs ../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl \
  -fsplit ../output/toy.dblp.v12.json/splits.json \
  -fpred ../output/toy.dblp.v12.json/bnn/ \
  -np_ratio 0.5 \
  -reranker det_cons \
  -output ../output/toy.dblp.v12.json/
```

Where the arguements are:

  > `fteamsvecs`: the sparse matrix representation of all teams in a pickle file, including the teams whose members are predicted in `--pred`. It should contain a dictionary of three `lil_matrix` with keys `[id]` of size `[#teams × 1]`, `[skill]` of size `[#teams × #skills]`, `[member]` of size `[#teams × #experts]`. Simply, each row of a metrix shows the occurrence vector of skills and experts in a team. For a toy example, try 
  ```
  import pickle
  with open(./data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl) as f: teams=pickle.load(f)
  ```
  
  > `fsplit`: the split.json file that indicates the index (rowid) of teams whose members are predicted in `--pred`. For a toy example, see [`output/toy.dblp.v12.json/splits.json`](output/toy.dblp.v12.json/splits.json)  

  > `fpred`: a file or folder that includes the prediction files of a neural team formation methods in the format of `torch.ndarray`. The file name(s) should be `*.pred` and the content is `[#test × #experts]` probabilities that shows the membership probability of an expert to a team in test set. For a toy example, try 
  ```
  import torch
  torch.load(./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/f0.test.pred)
  ```     

  > `np_ratio`: the desired `nonpopular` ratio among members of predicted teams after mitigation process by re-ranking algorithms. E.g., 0.5.
  
  > `reranker`: fairness-aware reranking algorithm from {`det_greedy`, `det_cons`, `det_relaxed`, `fa-ir`}. Eg. `det_cons`.  

  > `output`: the path to the reranked predictions of members for teams, as well as, the teams' success and fairness evaluation `before` and `after` mitigation process.

## 3. Pipeline
<p align="center"><img src='./misc/flow.png' width="1000" ></p>
 
`Adila` needs preprocessed information about the teams in the form of sparse matrix representation (`-fteamsvecs`) and neural team formation prediction file(s) (`-fpred`), obtained from [`OpeNTF`](https://github.com/fani-lab/OpeNTF/tree/main):

```bash
├── data
│   └── preprocessed
│       └── dblp
│           └── toy.dblp.v12.json
│               └── teamsvecs.pkl     #sparse matrix representation of teams
├── output
    └── toy.dblp.v12.json
        ├── bnn
        │   └── t31.s11.m13.l[100].lr0.1.b4096.e20.s1
        │       ├── f0.test.pred
        │       ├── f1.test.pred
        │       ├── f2.test.pred
        └── splits.json #rowids of team instances in n-fold train-valid splits, and a final test split
```

`Adila` has three main steps:

### 3.1. Labeling

Based on the distribution of experts on teams, which is power law (long tail) as shown in the figure, we label those in the `tail` as `nonpopular` and those in the `head` as popular. 
<p align="center"><img src='./misc/bias_ecir_23/latex/figures/nteams_candidate-idx_.png' width="250" ></p>

To find the cutoff between `head` and `tail`, we calculate the average number of teams per expert over the whole dataset. As seen in the table, this number is `62.45` and the popular/nonpopular ratio is `0.426/0.574`.  The result is a Boolean value in `{popular: True, nonpopular: False}` for each expert and is save in `{output}/popularity.csv` like [`./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/popularity.csv`](./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/popularity.csv) 
 
|             imdb                       |     |          |
|------------------------------------|:-------:|:--------:|
|                                    |   raw   | filtered |
| #movies                           | 507,034 |  32,059  |
| #unique casts and crews           | 876,981 |   2,011  |
| #unique genres                    |    28   |    23    |
| average #casts and crews per team |   1.88  |   3.98   |
| average #genres per team          |   1.54  |   1.76   |
| average #movie per cast and crew  |   1.09  |   62.45  |
| average #genre per cast and crew  |   1.59  |   10.85  |
| #team w/ single cast and crew     | 322,918 |     0    |
| #team w/ single genre             | 315,503 |  15,180  |
  
`Future:` We will consider equal area under the curve for the cutoff.

### 3.2. Gender
The following figures will demonstrate the gender distributions in `imdb`, `dblp` and `uspt`  datasets.
<p align="center">
 <img src='./misc/imdb_nmembers_nteams_regular_edited.png' width="240" >
 <img src='./misc/dblp_nmembers_nteams_regular_edited.png' width="250" >
 <img src='./misc/uspt_nmembers_nteams_regular_edited.png' width="225" >
</p>

### 3.3. Reranking 
  
We apply rerankers from [`deterministic greedy re-ranking methods [Geyik et al. KDD'19]`](https://dl.acm.org/doi/10.1145/3292500.3330691), including `{'det_greedy', 'det_cons', 'det_relaxed'}` to mitigate `populairty bias`. The reranker needs a cutoff `k_max` which is set to `10` by default. 

The result of predictions after reranking is saved in `{output}/rerank/{fpred}.{reranker}.{k_max}.rerank.pred` like [`./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f0.test.pred.det_cons.10.rerank.pred`](./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f0.test.pred.det_cons.10.rerank.pred) .

### 3.4. Evaluations 
  
We evaluate `fairness` and `utility` metrics `before` and `after` applying rerankers on team predictions to answer two research questions (RQs):
    
**`RQ1:`** Do state-of-the-art neural team formation models produce fair teams of experts in terms of popularity bias? To this end, we measure the fairness scores of predicted teams `before` applying rerankers. 
    
**`RQ2:`** Do state-of-the-art deterministic greedy re-ranking algorithms improve the fairness of neural team formation models while maintaining their accuracy? To this end, we measure the `fairness` and `utility` metrics `before` and `after` applying rerankers.
    
The result of `fairness` metrics `before` and `after` will be stored in `{output}.{algorithm}.{k_max}.{faireval}.csv` like [`./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f2.test.pred.det_cons.10.faireval.csv`](./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f2.test.pred.det_cons.10.faireval.csv) .
    
The result of `utility` metrics `before` and `after` will be stored in `{output}.{algorithm}.{k_max}.{utileval}.csv` like [`./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f1.test.pred.det_cons.10.utileval.csv`](./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f1.test.pred.det_cons.10.utileval.csv).
   
`Future:` We will consider other fairness metrics.

After successful run of all steps, [`./output`](./output) contains:

```bash
├── output
    └── toy.dblp.v12.json
        ├── bnn
        │   └── t31.s11.m13.l[100].lr0.1.b4096.e20.s1
        │       ├── f0.test.pred
        │       ├── f1.test.pred
        │       ├── f2.test.pred
        │       └── rerank/{popularity, gender}
        │           ├── f0.test.pred.det_cons.10.faireval.csv
        │           ├── f0.test.pred.det_cons.10.utileval.csv
        │           ├── f0.test.pred.det_cons.10.rerank.csv
        │           ├── f0.test.pred.det_cons.10.rerank.pred
        │           ├── f1.test.pred.det_cons.10.faireval.csv
        │           ├── f1.test.pred.det_cons.10.utileval.csv
        │           ├── f1.test.pred.det_cons.10.rerank.csv
        │           ├── f1.test.pred.det_cons.10.rerank.pred
        │           ├── f2.test.pred.det_cons.10.faireval.csv
        │           ├── f2.test.pred.det_cons.10.utileval.csv
        │           ├── f2.test.pred.det_cons.10.rerank.csv
        │           ├── f2.test.pred.det_cons.10.rerank.pred
        │           ├── labels.csv
        │           ├── rerank.time
        │           └── stats.pkl
        └── splits.json
```

## 4. Result
Our results show that although we improve fairness significantly, our utility metric drops extensively. Part of this phenomenon is described in [`Fairness in Ranking, Part I: Score-Based Ranking [Zehlike et al. ACM Computing Surveys'22]`](https://dl.acm.org/doi/full/10.1145/3533379). When we apply representation constraints on individual attributes, like race , popularity and gender and we want to maximize a score with respect to these constraints, utility loss can be particularly significant in historically disadvantaged intersectional groups. The following tables contain the results of our experiments on the `bnn`, `bnn_emb` and `random` baselines with `greedy`, `conservative` and `relaxed` re-ranking algorithms with `demographic parity` fairness notion.
| [``bnn(3.8 GB)``](https://uwin365-my.sharepoint.com/:f:/g/personal/ghasrlo_uwindsor_ca/Ej41Qn2GHytKhKpbyuiwLCABgUFOll74nBndOxQbDnLVMA?e=WtzNpd) |         |                |             |                |             |                |                      |
|:-----------------------------------------------------:|:-------:|:--------------:|:-----------:|:--------------:|:-----------:|:--------------:|:--------------------:|
|                                                       |         |     greedy     |             |  conservative  |             |     relaxed    |                      |
|                                                       |  before | after | $\Delta$ | after | $\Delta$ | after | $\Delta$ |
|                ndcg2 &uarr;               | 0.695% |     0.126%    |   -0.569%  |     0.091%    |   -0.604%  |     0.146%    |       -0.550%       |
|                ndcg5 &uarr;               | 0.767% |     0.141%    |   -0.626%  |     0.130%    |   -0.637%  |     0.130%    |       -0.637%       |
|               ndcg10 &uarr;               | 1.058% |     0.247%    |   -0.811%  |     0.232%    |   -0.826%  |     0.246%    |       -0.812%       |
|                map2 &uarr;                | 0.248% |     0.060%    |   -0.188%  |     0.041%    |   -0.207%  |     0.063%    |       -0.185%       |
|                map5 &uarr;                | 0.381% |     0.083%    |   -0.298%  |     0.068%    |   -0.313%  |     0.079%    |       -0.302%       |
|                map10 &uarr;               | 0.467% |     0.115%    |   -0.352%  |     0.101%    |   -0.366%  |     0.115%    |       -0.352%       |
|               ndlkl &darr;             |  0.2317 |     0.0276     |   -0.2041   |     0.0276     |   -0.2041   |     0.0273     |        -0.2043       |

| [``bnn_emb(3.79 GB)``](https://uwin365-my.sharepoint.com/:f:/g/personal/ghasrlo_uwindsor_ca/El75TMyU4D1Dt39_yLacGxYBSF2a4ntnyiZ7vq4rLy8dCg?e=skz450) |         |                |             |                |             |                |                      |
|:----------------------------------------------------------:|:-------:|:--------------:|:-----------:|:--------------:|:-----------:|:--------------:|:--------------------:|
|                                                            |         |     greedy     |             |  conservative  |             |     relaxed    |                      |
|                                                            |  before | after | $\Delta$ | after | $\Delta$ | after | $\Delta$ |
|                  ndcg2 &uarr;                  | 0.921% |     0.087%    |   -0.834%  |     0.121%    |   -0.799%  |     0.087%    |       -0.834%       |
|                  ndcg5 &uarr;                  | 0.927% |     0.117%    |   -0.810%  |     0.150%    |   -0.777%  |     0.117%    |       -0.810%       |
|                  ndcg10 &uarr;                 | 1.266% |     0.223%    |   -1.043%  |     0.241%    |   -1.025%  |     0.223%    |       -1.043%       |
|                  map2 &uarr;                  | 0.327% |     0.034%    |   -0.293%  |     0.057%    |   -0.270%  |     0.034%    |       -0.293%       |
|                  map5 &uarr;                  | 0.469% |     0.059%    |   -0.410%  |     0.084%    |   -0.386%  |     0.059%    |       -0.410%       |
|                  map10 &uarr;                  | 0.573% |     0.093%    |   -0.480%  |     0.111%    |   -0.461%  |     0.093%    |       -0.480%       |
|                  ndkl &darr;                |  0.2779 |     0.0244     |   -0.2535   |     0.0244     |   -0.2535   |     0.0241     |        -0.2539       |

|           [``random(2.41 GB)``](https://uwin365-my.sharepoint.com/:f:/g/personal/ghasrlo_uwindsor_ca/EkTgR0AjvIpNvz0Vsu-JwwoBMxl4kJsZxJBUI0zdQUxcTw?e=VYC66y)          |          |                |             |                |             |                |                      |
|:-------------------------:|:--------:|:--------------:|:-----------:|:--------------:|:-----------:|:--------------:|:--------------------:|
|                           |          |     greedy     |             |  conservative  |             |     relaxed    |                      |
|                           |  before  | after | $\Delta$ | after | $\Delta$ | after | $\Delta$ |
|  ndcg2 &uarr;  | 0.1711% |     0.136%    |   -0.035%  |     0.205%    |   0.034%   |     0.205%    |        0.034%       |
|  ndcg5 &uarr;  | 0.1809% |     0.170%    |   -0.011%  |     0.190%    |   0.009%   |     0.190%    |        0.009%       |
| ndcg10 &uarr;  | 0.3086% |     0.258%    |   -0.051%  |     0.283%    |   -0.026%  |     0.283%    |       -0.026%       |
|  map2 &uarr;   | 0.0617% |     0.059%    |   -0.003%  |     0.089%    |   0.028%   |     0.089%    |        0.028%       |
|  map5 &uarr;   | 0.0889% |     0.095%    |   0.006%   |     0.110%    |   0.021%   |     0.110%    |        0.021%       |
|  map10 &uarr;  | 0.1244% |     0.121%    |   -0.003%  |     0.140%    |   0.016%   |     0.140%    |        0.016%       |
| ndkl &darr; |  0.0072  |     0.0369     |    0.0296   |     0.0366     |    0.0293   |     0.0366     |        0.0294        |

The files containing the rest of our experiment results with various notions, datasets ,and algorithms are as follows:

|   | file |
|---|------|
| 1 |  [Demographic Parity.Popularity.Conservative.DBLP](https://uwin365.sharepoint.com/:u:/s/cshfrg-FairTeamFormation/EdAP6vdC5DhFri_3_-Km9XkBBlQQxHM90lPdjpB6wMLnfA?e=cJez2A)    |
| 2 |    [Demographic Parity.Popularity.Greedy.DBLP](https://uwin365.sharepoint.com/:u:/s/cshfrg-FairTeamFormation/EYsfhT46eDhLhv-TknKBxXgBhehSy-0aZM9KVSLS2g_eZw?e=K22gNK)    |
| 3 |   [Demographic Parity.Popularity.Relaxed.DBLP](https://uwin365.sharepoint.com/:u:/s/cshfrg-FairTeamFormation/EQOOClnM_h9Nlv7R35qnZ3EBP_2OeSQlzMGwcYLakrGmoA?e=Z98DXr)     |
| 4 |   [Equality of Opportunity.Popularity.Greedy.IMDB](https://uwin365.sharepoint.com/:u:/s/cshfrg-FairTeamFormation/EcG2O9_G1BdMvq5a4DPxSKQBZyZnRd_ClsZL_WEp471NCw?e=vf4oV6)     |
| 5 |   [Equality of Opportunity.Popularity.Conservative.IMDB](https://uwin365.sharepoint.com/:u:/s/cshfrg-FairTeamFormation/EfleYepwdkJFg01oWc4v4BkBty5oslXOOjhbqkIS4nGUaA?e=fuaXEQ)     |
| 6 |    [Equality of Opportunity.Popularity.Relaxed.IMDB](https://uwin365.sharepoint.com/:u:/s/cshfrg-FairTeamFormation/EaCptkFA-WhJs9Dn7BT2FEIBDALS633h92NRMgU7ODIUZA?e=By2tf3)    |
| 7 |    [Equality of Opportunity.Popularity.Greedy.DBLP](https://uwin365.sharepoint.com/:u:/s/cshfrg-FairTeamFormation/ESgfU0Fh7pNKu7puaAEZhZYBq1WGZqrUJte4cYXF9-MVSQ?e=fDLBFI)    |
| 8 |    [Equality of Opportunity.Popularity.Relaxed.DBLP](https://uwin365.sharepoint.com/:u:/s/cshfrg-FairTeamFormation/EWqcV2vMV0NPq2mnoRHSn8ABi2ex1-KTe4Gja5dibsT0Hw?e=ZXnWfd)    |
| 9 |    [Equality of Opportunity.Popularity.Conservative.DBLP](https://uwin365.sharepoint.com/:u:/s/cshfrg-FairTeamFormation/ER7epKNxKxNFs-mPlJRuWi4B3C8IfnQLiQ72N-TXRdOdUQ?e=tIam6x)    |

## 5. Acknowledgement
We benefit from [``pytrec``](https://github.com/cvangysel/pytrec_eval), [``reranking``](https://github.com/yuanlonghao/reranking), and other libraries. We would like to thank the authors of these libraries and helpful resources.
  
## 6. License
©2024. This work is licensed under a [CC BY-NC-SA 4.0](license.txt) license.

## 7. Citation
```
@inproceedings{DBLP:conf/bias/LoghmaniF23,
  author    = {Hamed Loghmani and Hossein Fani},
  title     = {Bootless Application of Greedy Re-ranking Algorithms in Fair Neural Team Formation},
  booktitle = {Advances in Bias and Fairness in Information Retrieval - Fourth International Workshop, {BIAS} 2023, Dublin, Irland, April 2, 2023, Revised Selected Papers},
  pages     = {108--118},
  publisher = {Springer Nature Switzerland},
  year      = {2023},
  url       = {https://doi.org/10.1007/978-3-031-37249-0_9},
  doi       = {10.1007/978-3-031-37249-0_9},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

