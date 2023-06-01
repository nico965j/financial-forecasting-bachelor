#!/bin/sh 
#BSUB -q gpuv100
#BSUB -J "model_6vars"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -M 4GB
### wall-time 4 hr
#BSUB -W 4:00 
#BSUB -u simondanielschneider@gmail.com
#BSUB -N

### Doesnt overwrite, just makes new. 
#BSUB -o outputs/GPU_Output_%J.out 
#BSUB -e errors/GPU_Error_%J.err

### Activate virtual environment
source act-venv.sh

module load cuda/11.0

### Write which script to run below
python3 train_eval.py