#!/bin/bash

#SBATCH --job-name=generate_pat_fill_data
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --account=wiensj1
#SBATCH --partition=standard

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10g


#execute code
cd #directory containing put_in_FIDDLE_form.py

python3 put_in_FIDDLE_form.py \
        --demographic_dataframes_path = "/home/jaewonh/data/dataframes/" \
        --patient_fill_info_path = "/home/jaewonh/data/" \
        --target_path = "/home/jaewonh/data/" \