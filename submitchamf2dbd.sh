#!/bin/bash
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=32g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:v100:1
#SBATCH -t 4999
#SBATCH -J wa-mrpre
#SBATCH -e jobs/error%A.err
#SBATCH -o jobs/out%A.out
#SBATCH -A PSYC0002
#SBATCH --mail-type=ALL
#SBATCH --mail-user=washbee1@student.gsu.edu
#SBATCH --oversubscribe

sleep 5s

export PATH=/data/users2/washbee/anaconda3/bin:$PATH
source /data/users2/washbee/anaconda3/etc/profile.d/conda.sh
conda activate /data/users2/washbee/anaconda3/envs/cham3d
python chamf2dbd.py

sleep 10s
