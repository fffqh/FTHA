#!/bin/bash
#

#SBATCH --job-name=detect
#SBATCH -N 1
#SBATCH -p L40
#SBATCH --output=detect_output.txt
#SBATCH --error=detect_errors.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load cuda/11.1
source activate yolov7
cd yolov7_main

for f in /share/home/tj14034/data/fqh/code/advTracker/yolov7_main/inference/multi/output_add_*/ ; do
srun python detect.py --source $f --classes 0 2 5 7
done