#!/bin/bash
#SBATCH --job-name=tabnet_xgb_job      # Descriptive job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks (usually 1 for single-node)
#SBATCH --cpus-per-task=10             # CPU cores for your process
#SBATCH --constraint=skylake           # Request a Skylake node (supports large memory)
#SBATCH --time=24:00:00                # Walltime: 24 hours (adjust as needed)
#SBATCH --mem=512G                     # 512 GB of RAM (adjust as needed)
#SBATCH --partition=batch              # Partition name (check your cluster)
#SBATCH --error=tabnet_xgb.%J.err      # Standard error file
#SBATCH --output=tabnet_xgb.%J.out     # Standard output file

# Move to your working directory
cd /ibex/user/hinkovn/TabNet

# Load the required Python module
module load python/3.9.16

# Optional: check which Python you have
which python

# Run your script
python size.py