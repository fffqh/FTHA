#!/bin/bash
#

#SBATCH --job-name=fcv
#SBATCH --output=f2v_output.txt
#SBATCH --error=f2v_errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=intel
#SBATCH --nodelist=cpui[12]
#SBATCH --mem=32G
#SBATCH --time=40:00:00

module load cuda/11.1
source activate yolov7
cd yolov7_main
for f in ./yolov7/inference/cars/output_add_*/ ; do
srun python frame2video.py --i $f --o $f'output_add.mp4' --f 2
done
