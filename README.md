# ``Adila``<sup>*</sup>: Fairness-Aware Team Formation
<sup>*[ عادلة, feminine Arabic given name, meaning just and fair](https://en.wikipedia.org/wiki/Adila_(name))<sup>

`Team Formation` aims to automate forming teams of experts who can successfully solve difficult tasks. While state-of-the-art neural team formation methods are able to efficiently analyze massive collections of experts to form effective collaborative teams, they largely ignore the fairness in recommended teams of experts. Fairness breeds innovation and increases teams' success by enabling a stronger sense of community, reducing conflict, and stimulating more creative thinking. In `Adila`, we study the application of `fairness-aware` team formation algorithms to mitigate the potential popularity bias in the neural team formation models. Our experiments show that, first, neural team formation models are biased toward popular experts. Second, although deterministic re-ranking algorithms mitigate popularity bias substantially, they severely hurt the efficacy of teams.

> We have studied the application of state-of-the-art [`deterministic greedy re-ranking methods [Geyik et al. KDD'19]`](https://dl.acm.org/doi/10.1145/3292500.3330691) to mitigate `populairty bias` based on `equality of opportunity` for state-of-the-art neural team formation methods from [`OpeNTF`](https://github.com/fani-lab/opeNTF/). Our experiments show that:
> - Neural team formation models are biased toward popular experts;
> - Although deterministic re-ranking algorithms mitigate popularity bias substantially, they severely hurt the efficacy of teams. 

> Currently, we are investigating:
> - Other fairness factors like demographic attributes, including age, race, and gender; 
> - Developing machine learning-based models using Learning-to-Rank (L2R) techniques to mitigate popularity bias as opposed to deterministic greedy algorithms.

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Pipeline](#3-pipeline)
4. [Result](#4-result)
5. [Acknowledgement](#5-acknowledgement)
6. [License](#6-license)  
7. [Citation](#7-citation)

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
To run `Adila`, you can use [./src/main.py](./src/main.py):

```bash
cd src
python -u main.py \
  -fteamsvecs ../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl \
  -fsplit ../output/toy.dblp.v12.json/splits.json \
  -fpred ../output/toy.dblp.v12.json/bnn/ \
  -ratio 0.5 \
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

  > `ratio`: the desired `nonpopular` ratio among members of predicted teams after mitigation process by re-ranking algorithms. E.g., 0.5.
  
  > `reranker`: fairness-aware reranking algorithm from {det_greedy, det_cons, det_relaxed}. Eg. det_cons'.  

  > `output`: the path to the reranked predictions of members for teams, as well as, the teams' success and fairness evaluation `before` and `after` mitigation process.

## 3. Pipeline

`Adila` has three steps:

1. Labeling: Based on the distribution of experts on teams, which is power law (long tail) as shown in the figure, we label those in the `tail` as `nonpopular` and those in the `head` as popular.
   To find the cutoff between `head` and `tail`, we calculate the average number of teams per expert over the whole dataset. As seen in table, this number is `62.45` and the popular/nonpopular ratio is `0.426/0.574`.  The result is a Boolean value in `{popular: True, nonpopular: False}` for each expert and is save in `{output}/popularity.csv` like [`./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/popularity.csv`](./output/toy.dblp.v12.json/bnn/t31.s11.m13.l[100].lr0.1.b4096.e20.s1/rerank/popularity.csv) 
    
    `Future:` We will consider equal area under the curve for the cutoff.
   
2. Reranking: We apply rerankers from [`deterministic greedy re-ranking methods [Geyik et al. KDD'19]`](https://dl.acm.org/doi/10.1145/3292500.3330691), including `{'det_greedy', 'det_cons', 'det_relaxed'}` to mitigate `populairty bias`. The reranker needs a cutoff `k_max` which is set to `10` by default. 
   The result of predictions after reranking is saved in `{output}/rerank/{fpred}.rerank.{reranker}.{k_max}` like ***.

3. Evaluations: We evaluate `fairness` and `utility` metrics `before` and `after` applying rerankers on team predictions to answer two research questions (RQs):
    
    **`RQ1:`** Do state-of-the-art neural team formation models produce fair teams of experts in terms of popularity bias? To this end, we measure the fairness scores of predicted teams `before` applying rerankers. 
    
    **`RQ2:`** Do state-of-the-art deterministic greedy re-ranking algorithms improve the fairness of neural team formation models while maintaining their accuracy? To this end, we measure the `fairness` and `utility` metrics `before` and `after` applying rerankers.
    
    The result of `fairness` metrics `before` and `after` will be stored in `{output}.{algorithm}.{k_max}.{faireval}.csv` like ***.
    
    The result of `utility` metrics `before` and `after` will be stored in `{output}.{algorithm}.{k_max}.{utileval}.csv` like ***.
   
    `Future:` We will consider other fairness metrics.

## 4. Result
***

## 5. Acknowledgement
We benefit from [``pytrec``](https://github.com/cvangysel/pytrec_eval), [``reranking``](https://github.com/yuanlonghao/reranking), and other libraries. We would like to thank the authors of these libraries and helpful resources.
  
## 6. License
©2023. This work is licensed under a [CC BY-NC-SA 4.0](license.txt) license.

Hamed Loghmani<sup>1</sup>, [Hossein Fani](https://hosseinfani.github.io/)<sup>1,2</sup> 

<sup><sup>1</sup>School of Computer Science, Faculty of Science, University of Windsor, ON, Canada.</sup>
<sup><sup>2</sup>[hfani@uwindsor.ca](mailto:hfani@uwindsor.ca)</sup>

## 7. Citation
```
@inproceedings{DBLP:conf/bias/LoghmaniF23,
  author    = {Hamed Loghmani and Hossein Fani},
  title     = {Bootless Application of Greedy Re-ranking Algorithms in Fair Neural Team Formation},
  booktitle = {Advances in Bias and Fairness in Information Retrieval - Fourth International Workshop, {BIAS} 2023, Dublin, Irland, April 2, 2023, Revised Selected Papers},
  publisher = {Springer},
  year      = {2023},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

