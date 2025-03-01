#!/bin/bash
#SBATCH --time=0-4:00:00
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

module load python
module load pytorch

DATASET=$1
METHOD=$2
CMDS="${@:3}"

cd ~/variationalsearch
source vsdenv/bin/activate
batch_rounds --dataset $DATASET --method $METHOD --device "cuda" $CMDS