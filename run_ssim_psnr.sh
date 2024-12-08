#!/bin/bash
#

#SBATCH --job-name=fcv
#SBATCH --output=ssim_output.txt
#SBATCH --error=ssim_errors.txt
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
srun python case_ssim_psnr.py \
    --r './yolov7/inference/cross/origin/' \
    --i './yolov7/inference/cross/output_mov_0.001_200/' \
    --o './yolov7/inference/cross/output_mov_0.001_200/'
