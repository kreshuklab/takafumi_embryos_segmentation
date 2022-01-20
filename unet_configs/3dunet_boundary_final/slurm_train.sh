#!/bin/bash

#SBATCH -A kreshuk
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 64G
#SBATCH -t 24:00:00
#SBATCH -o /g/kreshuk/wolny/workspace/dimitri_data/train.log
#SBATCH -e /g/kreshuk/wolny/workspace/dimitri_data/error.log
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=adrian.wolny@embl.de
#SBATCH -p gpu
#SBATCH -C "gpu=2080Ti|gpu=1080Ti"
#SBATCH --gres=gpu:1

module load cuDNN

export PYTHONPATH="/g/kreshuk/wolny/workspace/pytorch-3dunet:$PYTHONPATH"

/g/kreshuk/wolny/miniconda3/envs/pytorch-3dunet/bin/python /g/kreshuk/wolny/workspace/pytorch-3dunet/pytorch3dunet/train.py --config /g/kreshuk/wolny/workspace/dimitri_data/train_config.yml
