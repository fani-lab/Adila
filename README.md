# ``Adila``<sup>*</sup>: Fairness-Aware Team Formation
<sup>*[ A feminine Arabic given name, عادلة, meaning just and fair.](https://en.wikipedia.org/wiki/Adila_(name))<sup>

`Team Formation` aims to automate forming teams of experts who can successfully solve difficult tasks. While state-of-the-art neural team formation methods are able to efficiently analyze massive collections of experts to form effective collaborative teams, they largely ignore the fairness in recommended teams of experts. Fairness breeds innovation and increases teams' success by enabling a stronger sense of community, reducing conflict, and stimulating more creative thinking. In `Adila`, we study the application of `fairness-aware` team formation algorithms to mitigate the potential popularity bias in the neural team formation models. Our experiments show that, first, neural team formation models are biased toward popular experts. Second, although deterministic re-ranking algorithms mitigate popularity bias substantially, they severely hurt the efficacy of teams.

> We have studied the application of state-of-the-art [`deterministic greedy re-ranking methods`](https://dl.acm.org/doi/10.1145/3292500.3330691) to mitigate `populairty bias` based on `equality of opportunity` for state-of-the-art neural team formation methods from [`OpeNTF`](https://github.com/fani-lab/opeNTF/). 

> Currently, ...

> In future work, we will be studying ...

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Pipeline](#3-pipeline)
4. [Acknowledgement](#4-acknowledgement)
5. [License](#5-license)  

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
python -u main.py 
  --pred ../output/toy.dblp.v12.json/bnn/ \
  --fsplit ../output/toy.dblp.v12.json/splits.json \
  --fteamsvecs ../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl \
  --ratios 0.5 0.5 \
  --output ../output/toy.dblp.v12.json 
```

Where the arguements are:

> `pred`: a folder that includes the prediction files of neural methods in the format of ??? 
> `fsplit`: the split.json file that indicates the index (rowid) of teams whose members are predicted in `--pred`  
> `fteamsvecs`: the sparse matrix representation of all teams, including the teams whose members are predicted in `--pred`
> `ratios`: the desired `popular`/`nonpopular` ratio among members of predicted teams after mitigation process by re-ranking algorithms 
> `output`: the path to the re-ranked predictions of experts for teams, as well as, the teams' success and fairness evaluation `before` and `after` mitigation process.

## 3. Pipeline

???

## 4. Acknowledgement
We benefit from [``pytrec``](https://github.com/cvangysel/pytrec_eval), [``reranking``](https://github.com/yuanlonghao/reranking), and other libraries. We would like to thank the authors of these libraries and helpful resources.
  
## 5. License
©2023. This work is licensed under a [CC BY-NC-SA 4.0](license.txt) license.

Hamed Loghmani<sup>1</sup>, [Hossein Fani](https://hosseinfani.github.io/)<sup>1,2</sup> 

<sup><sup>1</sup>School of Computer Science, Faculty of Science, University of Windsor, ON, Canada.</sup>
<sup><sup>2</sup>[hfani@uwindsor.ca](mailto:hfani@uwindsor.ca)</sup>

## 6. Citation
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

