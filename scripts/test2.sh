#!/bin/bash

#SBATCH --job-name=Testing
#SBATCH -A CS156b
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres gpu:1
#SBATCH --mail-user=ecasanov@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu
#SBATCH --output=/groups/CS156b/2024/butters/%j.out
#SBATCH --error=/groups/CS156b/2024/butters/%j.err

source /Users/anush/CS156b-1/apaul_butters/bin/activate
# /home/ecasanov/CS156b/myenv/bin/activate
cd /groups/CS156b/2024/butters
python test1.py