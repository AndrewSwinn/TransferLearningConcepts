#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=ams90@kent.ac.uk
#SBATCH --array=1-29

source /home/bwc/ams90/miniconda3/etc/profile.d/conda.sh
conda activate torchenv

# Define your commands
COMMANDS=(
          "python ResnetConceptTrainer.py"
          "python ResnetConceptTrainer.py -c has_bill_shape"
          "python ResnetConceptTrainer.py -c has_wing_color"
          "python ResnetConceptTrainer.py -c has_upperparts_color"
          "python ResnetConceptTrainer.py -c has_underparts_color"
          "python ResnetConceptTrainer.py -c has_breast_pattern"
          "python ResnetConceptTrainer.py -c has_back_color"
          "python ResnetConceptTrainer.py -c has_tail_shape"
          "python ResnetConceptTrainer.py -c has_upper_tail_color"
          "python ResnetConceptTrainer.py -c has_head_pattern"
          "python ResnetConceptTrainer.py -c has_breast_color"
          "python ResnetConceptTrainer.py -c has_throat_color"
          "python ResnetConceptTrainer.py -c has_eye_color"
          "python ResnetConceptTrainer.py -c has_bill_length"
          "python ResnetConceptTrainer.py -c has_forehead_color"
          "python ResnetConceptTrainer.py -c has_under_tail_color"
          "python ResnetConceptTrainer.py -c has_nape_color"
          "python ResnetConceptTrainer.py -c has_belly_color"
          "python ResnetConceptTrainer.py -c has_wing_shape"
          "python ResnetConceptTrainer.py -c has_size"
          "python ResnetConceptTrainer.py -c has_shape"
          "python ResnetConceptTrainer.py -c has_back_pattern"
          "python ResnetConceptTrainer.py -c has_tail_pattern"
          "python ResnetConceptTrainer.py -c has_belly_pattern"
          "python ResnetConceptTrainer.py -c has_primary_color"
          "python ResnetConceptTrainer.py -c has_leg_color"
          "python ResnetConceptTrainer.py -c has_bill_color"
          "python ResnetConceptTrainer.py -c has_crown_color"
          "python ResnetConceptTrainer.py -c has_wing_pattern")

num_jobs=${#COMMANDS[@]}


# Set up an index for array job
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Execute the command
${COMMANDS[$INDEX]}
