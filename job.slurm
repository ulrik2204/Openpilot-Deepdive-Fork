#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --job-name=openpilot-deepdive
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpu:8
#SBATCH --constraint=gpu40g
#SBATCH --time=5-06:00:00
#SBATCH --output=output-deepdive.txt



# original time: 5-06:00:00
# load the required python module
module load Python/3.9.6-GCCcore-11.2.0

# Change to the work dir
cd /cluster/home/ulrikro/code/Openpilot-Deepdive-Fork

source venv/bin/activate
pip install -r requirements.txt

PORT=23333 srun python main.py --optimizer adamw
#PORT=23333 srun python main.py --resume=/path/to/checkpoint

# unload the required python module
module unload Python/3.9.6-GCCcore-11.2.0