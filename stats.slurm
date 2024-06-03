#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --job-name=openpilot-deepdive-stats
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40G
#SBATCH --time=0-04:00:00
#SBATCH --output=output-deepdive-stats.txt



module load Python/3.9.6-GCCcore-11.2.0

# Change to the work dir
cd /cluster/home/ulrikro/code/Openpilot-Deepdive-Fork

source venv/bin/activate
pip install -r requirements.txt

PORT=23333 srun python comma2k19_stats.py
#PORT=23333 srun python main.py --resume=/path/to/checkpoint

# unload the required python module
module unload Python/3.9.6-GCCcore-11.2.0