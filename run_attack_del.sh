#!/bin/bash
#

#SBATCH --job-name=del
#SBATCH -N 1
#SBATCH -p L40
#SBATCH --output=attack_del_output.txt
#SBATCH --error=attack_del_errors.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load cuda/11.1
source activate yolov7
cd yolov7
srun python attack_del.py --cases rain --eps 0.002 0.003 0.005 0.008 --rds 100 50 50 30
