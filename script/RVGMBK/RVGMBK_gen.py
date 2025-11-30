#!/usr/bin/env python3
"""
Generate PBS job scripts for RVGMBK parameter sweep.
This script generates all combinations of:
- beta: {0.01, 0.1, 1.0}
- margin: {0.5, 1.0, 2.0}
- num_prototypes: {1, 3, 5, 8}

Also generates:
- Markdown file to track experiment parameters
- submit_all.sh script to submit all jobs
"""

import os
from itertools import product
import datetime

# Parameter ranges
betas = [0.01, 0.1, 1.0]
margins = [0.5, 1.0, 2.0]
num_prototypes_list = [1, 3, 5, 8]

# Starting experiment ID
START_ID = 1

# Fixed parameters
linear_epochs = 50
finetune_epochs = 50
ngpus = 1
ncpus = 12
mem = "32GB"
walltime = "10:00:00"

# PBS script template
script_template = """#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus={ngpus}            
#PBS -l ncpus={ncpus}            
#PBS -l mem={mem}           
#PBS -l walltime={walltime}  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2

source /scratch/rp06/sl5952/Zelpha/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/Zelpha/.cache"
export HF_HUB_OFFLINE=1

cd ../..
# Run training with {exp_id}: num_prototypes={num_prototypes}, beta={beta}, margin={margin}
python3 src/train.py \\
    --model_name repvgg_a0.rvgg_in1k \\
    --num_prototypes {num_prototypes} \\
    --beta {beta} \\
    --margin {margin} \\
    --image_size 224 \\
    --linear_epochs {linear_epochs} --finetune_epochs {finetune_epochs} >> logs/{exp_id}.log 2>&1
"""

def generate_scripts():
    """Generate all parameter sweep scripts."""
    # Create output directory if it doesn't exist
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all combinations
    combinations = list(product(num_prototypes_list, betas, margins))
    
    print(f"Generating {len(combinations)} PBS scripts...")
    
    # Prepare data for markdown and submit script
    experiments = []
    submit_commands = []
    
    for idx, (num_prototypes, beta, margin) in enumerate(combinations, start=START_ID):
        # Format experiment ID
        exp_id = f"RVGMBK{idx:03d}"
        
        # Format script name
        script_name = f"{exp_id}.sh"
        script_path = os.path.join(output_dir, script_name)
        
        # Fill in template
        script_content = script_template.format(
            exp_id=exp_id,
            ngpus=ngpus,
            ncpus=ncpus,
            mem=mem,
            walltime=walltime,
            num_prototypes=num_prototypes,
            beta=beta,
            margin=margin,
            linear_epochs=linear_epochs,
            finetune_epochs=finetune_epochs
        )
        
        # Write script
        with open(script_path, 'w', newline='\n') as f:
            f.write(script_content)
        
        print(f"Created: {script_name}")
        
        # Store experiment info
        experiments.append({
            'id': exp_id,
            'num_prototypes': num_prototypes,
            'beta': beta,
            'margin': margin,
            'linear_epochs': linear_epochs,
            'finetune_epochs': finetune_epochs,
            'script': script_name
        })
        
        # Add to submit commands
        submit_commands.append(f"qsub {script_name}")
    
    # Generate markdown documentation
    generate_markdown(experiments)
    
    # Generate submit_all.sh script
    generate_submit_script(submit_commands)
    
    print(f"\nTotal scripts generated: {len(combinations)}")
    print(f"Markdown documentation: RVGMBK_experiments.md")
    print(f"Submit script: submit_all.sh")
    print("\nTo submit all jobs, run: bash submit_all.sh")

def generate_markdown(experiments):
    """Generate markdown documentation for all experiments."""
    md_content = """# RVGMBK Parameter Sweep Experiments

## Overview
This document tracks all RVGMBK parameter sweep experiments.

**Total Experiments:** {}

**Parameter Ranges:**
- `num_prototypes`: {{1, 3, 5, 8}}
- `beta`: {{0.01, 0.1, 1.0}}
- `margin`: {{0.5, 1.0, 2.0}}
- `linear_epochs`: 50
- `finetune_epochs`: 30

## Experiment Table

| Exp ID | num_prototypes | beta | margin | linear_epochs | finetune_epochs | robust_acc |
|--------|----------------|------|--------|---------------|-----------------|---|
""".format(len(experiments))
    
    for exp in experiments:
        md_content += "| {} | {} | {} | {} | {} | {} | TBD |\n".format(
            exp['id'],
            exp['num_prototypes'],
            exp['beta'],
            exp['margin'],
            exp['linear_epochs'],
            exp['finetune_epochs'],
        )
    
    md_content += """

## Notes
Update the Status and Results columns as experiments complete.

---
*Generated on: {}*
""".format(datetime.date.today().isoformat())
    
    with open("RVGMBK_experiments.md", 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print("Generated: RVGMBK_experiments.md")

def generate_submit_script(commands):
    """Generate submit_all.sh script."""
    submit_content = """#!/bin/bash
# Submit all RVGMBK parameter sweep jobs

echo "Submitting {} RVGMBK experiments..."
echo ""

""".format(len(commands))
    
    for cmd in commands:
        submit_content += f"{cmd}\n"
    
    submit_content += """
echo ""
echo "All jobs submitted!"
echo "Check status with: qstat -u $USER"
"""
    
    with open("submit_all.sh", 'w', newline='\n') as f:
        f.write(submit_content)
    
    print("Generated: submit_all.sh")

if __name__ == "__main__":
    generate_scripts()
