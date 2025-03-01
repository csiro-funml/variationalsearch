#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --mem=32gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

module load python
module load pytorch

CMDS="${@:1}"

cd ~/variationalsearch
source vsdenv/bin/activate
ehrlich --device cuda $CMDS