#!/bin/bash

##setup the env
#module load python/3.8;
source ../ENV/bin/activate;

##SBATCH --time=5:00:00
##SBATCH --account=def-hfani
##SBATCH --gpus-per-node=0
##SBATCH --mem=64000M
##SBATCH --mail-user=ghasrlo@uwindsor.ca
##SBATCH --mail-type=ALL
##nvidia-smi

python main.py