#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=32GB           
#PBS -l walltime=10:00:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/Zelpha/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/Zelpha/.cache"
export HF_HUB_OFFLINE=1

cd ../..
# Run training with T007
python3 src/train.py \
    --model_name eva_giant_patch14_224.clip_ft_in1k \
    --image_size 224 \
    --linear_epochs 50 --finetune_epochs 30 >> logs/T007.log 2>&1