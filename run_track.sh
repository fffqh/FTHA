#!/bin/bash
#

#SBATCH --job-name=track
#SBATCH -N 1
#SBATCH -p L40
#SBATCH --output=track_output.txt
#SBATCH --error=track_errors.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load cuda/11.1
source activate yolov7
cd yolov7_strongSORT

for f in /share/home/tj14034/data/fqh/code/advTracker/yolov7_main/inference/cars/output_add_0.001_*/ ; do
srun python track.py --source $f --save-vid --classes 0 2 5 7
done

for f in /share/home/tj14034/data/fqh/code/advTracker/yolov7_main/inference/cars/output_add_0.002_*/ ; do
srun python track.py --source $f --save-vid --classes 0 2 5 7
done

for f in /share/home/tj14034/data/fqh/code/advTracker/yolov7_main/inference/cars/output_add_0.005_*/ ; do
srun python track.py --source $f --save-vid --classes 0 2 5 7
done

for f in /share/home/tj14034/data/fqh/code/advTracker/yolov7_main/inference/cars/output_add_0.0005_*/ ; do
srun python track.py --source $f --save-vid --classes 0 2 5 7
done
