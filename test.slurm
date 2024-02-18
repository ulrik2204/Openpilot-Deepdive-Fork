#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --job-name=ulrikro-testing
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2 
#SBATCH --mem=5G
#SBATCH --output=hello.txt
#SBATCH --time=00:15:00



# original time: 5-06:00:00
# load the required python module
module load Python/3.9.6-GCCcore-11.2.0

# Change to the work dir 
cd /cluster/home/ulrikro/code/Openpilot-Deepdive-Fork

source op-dd-3.7/bin/activate

PORT=23333 srun python test.py

# unload the required python module
module unload Python/3.9.6-GCCcore-11.2.0