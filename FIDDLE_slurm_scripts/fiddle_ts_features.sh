#!/bin/bash

#SBATCH --job-name=make_ts
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=wiensj1
#SBATCH --partition=largemem

#SBATCH --time=27:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=15g

#remember to run this beforehand: module load python-anaconda3/5.2

#execute code
python3 -m /home/jaewonh/FIDDLE/FIDDLE.run \
    --data_path='/scratch/filip_root/filip/jaewonh/dt=60_no_bin/' \
    --input_fname='/scratch/filip_root/filip/jaewonh/fiddle_format_data.csv' \
    --population='/scratch/filip_root/filip/jaewonh/fiddle_format_pop.csv' \
    --T=365 \
    --dt=60 \
    --theta_1=0 \
    --theta_2=0 \
    --binarize=0 \
    --stats_functions 'sum'