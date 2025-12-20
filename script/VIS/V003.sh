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
# Run training with V003
python3 -m src.vis \
  --ckpt_zelpha outputs/V001/best_proto.pt \
  --ckpt_base   outputs/V002/best_proto.pt \
  --dataset UC_Merced \
  --batch_size 64 \
  --image_size 256 \
  --model_name zelpha \
  --output_dir outputs/V003 \
  --do_tsne --tsne_all --do_margin --do_scale_vis >> logs/V003.log 2>&1