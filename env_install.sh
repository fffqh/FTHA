#!/bin/bash
#

#SBATCH --job-name=laod
#SBATCH -N 1
#SBATCH -p L40
#SBATCH --output=env_output.txt
#SBATCH --error=env_errors.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load cuda/11.1
source activate yolov7
pip install pyyaml==6.0.1
pip install tqdm
pip install ipython psutil thop