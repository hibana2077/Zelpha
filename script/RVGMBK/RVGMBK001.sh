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
# Run training with RVGMBK001: num_prototypes=1, beta=0.01, margin=0.5
python3 src/train.py \
    --model_name repvgg_a0.rvgg_in1k \
    --num_prototypes 1 \
    --beta 0.01 \
    --margin 0.5 \
    --image_size 224 \
    --linear_epochs 50 --finetune_epochs 50 >> logs/RVGMBK001.log 2>&1
