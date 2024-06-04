#!/bin/bash

#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --partition=gpu
#SBATCH --mem=2G   # memory per CPU core
#SBATCH --time=00:10:00  
#SBATCH --output=/groups/CS156b/2024/butters/%j.out
#SBATCH --error==/groups/CS156b/2024/butters/%j.err
#SBATCH --partition=gpu
#SBATCH -A CS156b

#SBATCH -J "Test"   # job name
#SBATCH --mail-user=ecasanov@caltech.edu  # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source /Users/anush/CS156b-1/apaul_butters/bin/activate
# /home/ecasanov/CS156b/myenv/bin/activate

cd /groups/CS156b/2024/butters

python test1.py



