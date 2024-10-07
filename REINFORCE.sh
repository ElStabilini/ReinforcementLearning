#!/bin/bash

# SBATCH directives (optional)
#SBATCH --job-name = REINFORCE  # Job name
#SBATCH --nodelist = meitner            # Node to use

python3 training_REINFORCE.py