# Fairness Aware Team Formation
In classic team formation, our goal is to come up with a set of recommended experts for a specific problem or project. There are some necessary skills for each project and our experts must cover those while maximizing the chance of successfully finishing the project.
Most team formation approaches are not fairness aware. Hence, the results are usually biased. This bias might be on single or multiple protected attributes. For example, we might have a gender bias in the process of team formation or bias on the combination of gender and popularity. When we are planning to use these systems in industry and automation of real-life decision-making processes, we must make sure our system is fairness aware.
There are three groups of methodologies to address the bias phenomenon. The first group focus on the data-gathering process to obtain a fair representative dataset. The second group focus on model modification to train a fair model. Finally, the last group focus on post-prediction methods. These approaches re-rank the given predictions by the model in order to make it fair or at least less unfair.
At the moment our research focuses on the third group of methodologies, and we are trying to implement the techniques presented in “Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search” on our team formation framework namely, OpeNTF.


## Setup
using the [starter.ipynb](./starter.ipynb). Basic setup to run our project.

## results
Metric  | Mean
------------- | -------------
P_2  | 0.400000000000000
P_5  | 0.360000000000000
P_10  | 0.240000000000000
recall_2  | 0.316666666666667
recall_5  | 0.716666666666667
recall_10  | 0.950000000000000
ndcg_cut_2  | 0.309482245787633
ndcg_cut_5  | 0.497449381433493
ndcg_cut_10  | 0.605848208191883
map_cut_2  | 0.158333333333333
map_cut_5  | 0.352500000000000
map_cut_10  | 0.436626984126984
aucroc  | 0.678254437869823

## Future work

Running it on the final DBLP test dataset.