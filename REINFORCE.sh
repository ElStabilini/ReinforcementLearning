#!/bin/bash

# SBATCH directives (optional)
#SBATCH --job-name = REINFORCE  # Job name
#SBATCH --nodelist = meitner, lorentz, levicivita, dirac, poincare, turing, penrose, weyl, heaviside            # Node to use

python3 training_REINFORCE.py
