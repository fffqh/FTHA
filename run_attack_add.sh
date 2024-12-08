#!/bin/bash
#

#SBATCH --job-name=add
#SBATCH -N 1
#SBATCH -p L40
#SBATCH --output=attack_add_output.txt
#SBATCH --error=attack_add_errors.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load cuda/11.1
source activate yolov7
cd yolov7_main
srun python attack_add.py --cases cars --eps 0.005 0.005 0.005 0.005 --rds 20 40 60 80
