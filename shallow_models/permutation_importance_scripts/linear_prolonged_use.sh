#!/bin/bash

#SBATCH --job-name=lin_pu_imp
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=wiensj1
#SBATCH --partition=standard

#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10g


#execute code
cd #directory containing permutation_importances_no_bootstrapping.py

python3 permutation_importances_no_bootstrapping.py \
        --model_type='linear_svc' \
        --outcome='pu' \