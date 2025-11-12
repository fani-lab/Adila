# ``Adila``<sup>*</sup>: Fairness-Aware Team Recommendation
<sup>*[ عادلة, feminine Arabic given name, meaning just and fair](https://en.wikipedia.org/wiki/Adila_(name))<sup>

> Bootless Application of Greedy Re-ranking Algorithms in Fair Neural Team Formation. BIAS-ECIR, 2023.[`pdf`](https://hosseinfani.github.io/res/papers/2023_BIAS_ECIR_Bootless_Application_of_Greedy_Re-ranking_Algorithms_in_Fair_Neural_Team_Formation.pdf) [`doi`](https://doi.org/10.1007/978-3-031-37249-0_9) [`reviews`](https://hosseinfani.github.io/res/papers/2023_BIAS_ECIR_Bootless_Application_of_Greedy_Re-ranking_Algorithms_in_Fair_Neural_Team_Formation.txt) [`video`](https://www.youtube.com/watch?v=EmAFvANqzBM)

> A Probabilistic Greedy Attempt to be Fair in Neural Team Recommendation. COIN, 2025. `Under Review`

`Team Recommendation` aims to automate forming teams of experts who can collaborate and successfully solve tasks. While state-of-the-art methods are able to efficiently analyze massive collections of experts to recommend effective collaborative teams, they largely ignore the fairness in the recommended experts; our experiments show that they are biased toward `popular` and `male` experts. In `Adila`, we aim to mitigate the potential biases for fair team recommendation. Fairness breeds innovation and increases teams' success by enabling a stronger sense of community, reducing conflict, and stimulating more creative thinking. 

We have studied the application of state-of-the-art [`deterministic greedy re-ranking methods [Geyik et al. KDD'19]`](https://dl.acm.org/doi/10.1145/3292500.3330691) in addition to [`probabilistic greedy re-ranking methods [Zehlike et al. IP&M'22]`](https://dl.acm.org/doi/abs/10.1016/j.ipm.2021.102707)to mitigate `populairty bias` and `gender bias` based on `equality of opportunity` and `demographic parity` notions of fairness for state-of-the-art neural team formation methods from [`OpeNTF`](https://github.com/fani-lab/opeNTF/). Our experiments show that:
> Although deterministic re-ranking algorithms mitigate `popularity` xor `gender` bias, they hurt the efficacy of teams, i.e., higher fairness metrics yet lower utility metrics (successful team)

> Probabilistic greedy re-ranking algorithms mitigate `popularity` bias significantly and maintain utility. Though in terms of `gender`, such algorithms fail due to extreme bias in a dataset. 

Currently, we are investigating:
> Other fairness factors like demographic attributes, including `age`, and `race`; 

> Developing machine learning-based models using Learning-to-Rank (L2R) techniques to mitigate bias as opposed to deterministic greedy algorithms.

- [1. Setup](#1-setup)
- [2. Quickstart](#2-quickstart)
- [3. Pipeline](#3-pipeline)
  * [3.1. Labeling](#31-labeling)
  * [3.2. Gender Distribution](#32-gender)
  * [3.3. Reranking](#33-reranking)
  * [3.4. Evaluations](#34-evaluations)
- [4. Acknowledgement](#4-acknowledgement)
- [5. License](#5-license)

## 1. Setup & Quickstart
`Adila` needs ``Python=3.8`` and others packages listed in [``requirements.txt``](requirements.txt). By ``pip``, clone the codebase and install the required packages:

```sh
git clone https://github.com/Fani-Lab/Adila
cd Adila
pip install -r requirements.txt
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

Where the arguments are:

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
<p align="center"><img src='./docs/flow.png' width="500" ></p>
 
`Adila` needs preprocessed information about the teams in the form of sparse matrix representation (`-fteamsvecs`) and neural team formation prediction file(s) (`-fpred`), obtained from [`OpeNTF`](https://github.com/fani-lab/OpeNTF/tree/main):

```bash
.
├── data
│   └── {dblp, imdb, uspt}
└── output
    └── dblp
        └── toy.dblp.v12.json
            ├── gender.csv
            ├── indexes.pkl
            ├── splits.json
            ├── teams.pkl
            ├── teamsvecs.pkl
            └── bnn
                └── t31.s11.m13.l[100].lr0.1.b4096.e20.s1
                    ├── f0.test.pred
                    ├── f0.test.pred.eval.mean.csv
```

`Adila` has three main steps:

### 3.1. Popularity
<p align="center"><img src='./docs/bias_ecir_23/latex/figures/nteams_candidate-idx_.png' width="200" ></p>

Based on the distribution of experts on teams, which is power law (long tail) as shown in the figure, we label those in the `tail` as `nonpopular` and those in the `head` as popular. To find the cutoff between `head` and `tail`, we calculate the `avg` number of teams per expert over the entire dataset, or based on equal area under the curve `auc`. The result is a Boolean value in `{popular: True, nonpopular: False}` for each expert and is save in `{output}/popularity.csv` like [`./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/popularity.csv`](./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/popularity.csv) 
 
### 3.2. Gender
The following figures will demonstrate the gender distributions in `imdb`, `dblp` and `uspt`  datasets.
<p align="center">
 <img src='./docs/imdb_nmembers_nteams_regular_edited.png' width="200" >
 <img src='./docs/dblp_nmembers_nteams_regular_edited.png' width="210" >
 <img src='./docs/uspt_nmembers_nteams_regular_edited.png' width="200" >
</p>

### 3.3. Reranking 
  
We apply rerankers from [`deterministic greedy re-ranking methods [Geyik et al. KDD'19]`](https://dl.acm.org/doi/10.1145/3292500.3330691), including `{'det_greedy', 'det_cons', 'det_relaxed'}` to mitigate `populairty bias`. The reranker needs a cutoff `k_max` which is set to `10` by default. 

The result of predictions after reranking is saved in `{output}/rerank/{fpred}.{reranker}.{k_max}.rerank.pred` like [`./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f0.test.pred.det_cons.10.rerank.pred`](./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f0.test.pred.det_cons.10.rerank.pred).

### 3.4. Evaluations 
  
We evaluate `fairness` and `utility` metrics `before` and `after` applying rerankers on team predictions to answer two research questions (RQs):
    
**`RQ1:`** Do state-of-the-art neural team formation models produce fair teams of experts in terms of popularity bias? To this end, we measure the fairness scores of predicted teams `before` applying rerankers. 
    
**`RQ2:`** Do state-of-the-art deterministic greedy re-ranking algorithms improve the fairness of neural team formation models while maintaining their accuracy? To this end, we measure the `fairness` and `utility` metrics `before` and `after` applying rerankers.
    
The result of `fairness` metrics `before` and `after` will be stored in `{output}.{algorithm}.{k_max}.{faireval}.csv` like [`./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f2.test.pred.det_cons.10.faireval.csv`](./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f2.test.pred.det_cons.10.faireval.csv) .
    
The result of `utility` metrics `before` and `after` will be stored in `{output}.{algorithm}.{k_max}.{utileval}.csv` like [`./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f1.test.pred.det_cons.10.utileval.csv`](./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/f1.test.pred.det_cons.10.utileval.csv).
   
After successful run of all steps, [`./output`](./output) contains:

```bash
└── output
    └── dblp
        └── toy.dblp.v12.json
            └── bnn
                └── t31.s11.m13.l[100].lr0.1.b4096.e20.s1
                    └── rerank
                        ├── gender
                        │   ├── dp
                        │   │   ├── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.10.{ndkl,skew}.faireval.csv
                        │   │   ├── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.10.rerank.{csv,pred}
                        │   │   └── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.10.utileval.csv
                        │   ├── eo
                        │   │   ├── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.10.{ndkl,skew}.faireval.csv
                        │   │   ├── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.10.rerank.{csv,pred}
                        │   │   └── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.10.utileval.csv
                        │   ├── labels.csv
                        │   ├── ratios.pkl
                        │   └── stats.pkl
                        └── popularity
                            ├── dp
                            │   ├── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.{auc,avg}.10.{ndkl,skew}.faireval.csv
                            │   ├── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.{auc,avg}.10.rerank.{csv,pred}
                            │   └── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.{auc,avg}.10.utileval.csv
                            ├── eo
                            │   ├── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.{auc,avg}.10.{ndkl,skew}.faireval.csv
                            │   ├── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.{auc,avg}.10.rerank.{csv,pred}
                            │   └── f0.test.pred.{det_cons,det_greedy,det_relaxed,fa-ir}.{auc,avg}.10.utileval.csv
                            ├── labels.csv
                            ├── ratios.pkl
                            └── stats.pkl

```

## 4. Acknowledgement
We benefit from [``pytrec``](https://github.com/cvangysel/pytrec_eval), [``reranking``](https://github.com/yuanlonghao/reranking), and other libraries. We would like to thank the authors of these libraries and helpful resources.
  
## 5. License
©2025. This work is licensed under a [CC BY-NC-SA 4.0](license.txt) license.


