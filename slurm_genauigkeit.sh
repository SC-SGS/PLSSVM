#!/bin/bash
#SBATCH --job-name="genauigkeit $1"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=vancraar@ipvs.uni-stuttgart.de         # specify an email address
#SBATCH --mail-type=ALL                                   # send email when job status change (start, end, abortion and etc.)
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclusive



function func {
	ev=$1
	echo $ev
	echo $(/usr/bin/time -f "%e" svm-train -q -t 0 -e $ev $data)
	svm-predict $tes $model out
	svm-train-gpu -q -t 0 -e $ev $data
	svm-predict $tes $model out
}

data="/home/vancraar/Documents/Bachelor-Code/Cpp/C-SVC/data_predict_2.txt"
tes="/home/vancraar/Documents/Bachelor-Code/Cpp/C-SVC/data_test_2.txt"
model="data_predict_2.txt.model"

func $1
#func 0.01
#func 0.001
#func 0.0001
#func 0.00001
#func 0.000001
#func 0.0000001
#func 0.00000001
#func 0.000000001
#func 0.0000000001
#func 0.00000000001
#func 0.000000000001
#func 0.0000000000001
#func 0.00000000000001