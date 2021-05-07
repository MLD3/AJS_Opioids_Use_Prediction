#!/bin/bash

#SBATCH --job-name=make_flat
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=wiensj1
#SBATCH --partition=standard

#SBATCH --time=27:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10g

#remember to run this beforehand: module load python-anaconda3/5.2

#execute code
cd /home/jaewonh/FIDDLE/

python3 -m FIDDLE.run \
    --data_path='/scratch/filip_root/filip/jaewonh/dt365_new/' \
    --input_fname='/scratch/filip_root/filip/jaewonh/fiddle_format_data.csv' \
    --population='/scratch/filip_root/filip/jaewonh/fiddle_format_pop.csv' \
    --T=365 \
    --dt=365 \
    --theta_1=0 \
    --theta_2=0 \
    --stats_functions 'sum'