#!/bin/sh 
#BSUB -q gpuv100
#BSUB -J "sent_full"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -M 20GB
### wall-time 24 hr
#BSUB -W 24:00 
#BSUB -u simondanielschneider@gmail.com
#BSUB -N

### Doesnt overwrite, just makes new. 
#BSUB -o outputs/sent_FULL_Output_%J.out 
#BSUB -e errors/sent_FULL_Error_%J.err

### Activate virtual environment
source act-venv.sh

module load cuda/11.6

### Write which script to run below
python3 simple_sentiment.py