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
cd #directory containing generate_pat_fill_data.py

python3 generate_pat_fill_data.py \ 
        --dataframe_path = "/home/jaewonh/data/dataframes/" \
        --quantity_path = "/home/jaewonh/data/flat/FIDDLE_output_quantity/" \
        --recency_path = "/home/jaewonh/data/flat/FIDDLE_output_recency/" \
        --target_path = '/home/jaewonh/data/flat/train_test/' \