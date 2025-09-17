#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --account=OD-239291

module load python
module load pytorch/2.7.1-py312-cu124-mpi
module load transformers

CMDS="${@:1}"

cd ~/variationalsearch
source vsdenv/bin/activate
ehrlich --device cuda $CMDS