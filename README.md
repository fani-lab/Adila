# Fairness Aware Team Formation

In classic team formation, our goal is to come up with a set of recommended experts for a specific problem or project. There are some necessary skills for each project and our experts must cover those while maximizing the chance of successfully finishing the project.
Most team formation approaches are not fairness aware. Hence, the results are usually biased. This bias might be on single or multiple protected attributes. For example, we might have a gender bias in the process of team formation or bias on the combination of gender and popularity. When we are planning to use these systems in industry and automation of real-life decision-making processes, we must make sure our system is fairness aware.
There are three groups of methodologies to address the bias phenomenon. The first group focus on the data-gathering process to obtain a fair representative dataset. The second group focus on model modification to train a fair model. Finally, the last group focus on post-prediction methods. These approaches re-rank the given predictions by the model in order to make it fair or at least less unfair.
At the moment our research focuses on the third group of methodologies, and we are trying to implement the techniques presented in “Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search” on our team formation framework namely, OpeNTF.


## 1.Setup
Our recent version is being developed with Python3.8. For running this project locally, first the repo should be cloned:
```bash
git clone https://github.com/fani-lab/fair_team_formation.git
```
after creating a Python virtual environment you can install the required libraries and frameworks by running this command:
```bash
cd fair_team_formation
pip install -r requirements.txt
```
## 2.Quick Start
To run the code, you can use [src/main.py](https://github.com/fani-lab/fair_team_formation/blob/main/src/main.py)
in order to run the code on a toy dataset from dblp, you can use the following commands:
```bash
cd src
python -u main.py --pred
"../output/toy.dblp.v12.json/bnn/"
--fsplit
"../output/toy.dblp.v12.json/splits.json"
--fteamsvecs
"../data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl"
--output
"../output/toy.dblp.v12.json"
--ratios
0.5
0.5
```
