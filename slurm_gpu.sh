#!/bin/bash
#SBATCH --job-name="generate"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=vancraar@ipvs.uni-stuttgart.de         # specify an email address
#SBATCH --mail-type=ALL                                   # send email when job status change (start, end, abortion and etc.)
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclusive
#SBATCH --nice

mkdir -p /scratch/vancraar/libsvm
cd /scratch/vancraar/libsvm

i=$((SLURM_ARRAY_TASK_ID / 16 + 1))
j=$((SLURM_ARRAY_TASK_ID % 16 + 1))


python /home/vancraar/Documents/Bachelor-Code/dataset.py $[2**i] $[2**j] $[2**i]x$[2**j].txt 

/home/vancraar/Documents/Bachelor-Code/Cpp/C-SVC/svm-train-gpu -t 0 -c 1 -e 0.00000001 $[2**i]x$[2**j].txt 2>> ~/Documents/SLURM/test_gpu_lin_2.time 
rm -f $[2**i]x$[2**j].txt.model

/home/vancraar/Documents/Bachelor-Code/Cpp/C-SVC/svm-train-gpu -t 1 -c 1 -e 0.00000001  $[2**i]x$[2**j].txt 2>> ~/Documents/SLURM/test_gpu_poly_2.time 
rm -f $[2**i]x$[2**j].txt.model

/home/vancraar/Documents/Bachelor-Code/Cpp/C-SVC/svm-train-gpu -t 2 -c 1 -e 0.00000001  $[2**i]x$[2**j].txt 2>> ~/Documents/SLURM/test_gpu_rad_2.time 
rm -f $[2**i]x$[2**j].txt.model

rm -f $[2**i]x$[2**j].txt 
