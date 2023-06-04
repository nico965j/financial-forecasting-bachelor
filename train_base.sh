#!/bin/sh 
#BSUB -q hpc
#BSUB -J "base"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
### wall-time 4 hr
#BSUB -W 4:00 
#BSUB -u simondanielschneider@gmail.com
#BSUB -N

### Doesnt overwrite, just makes new.
#BSUB -o outputs/CPU_Output_%J.out
#BSUB -e errors/CPU_Error_%J.err

conf_path=conf_base.yml

### Activate virtual environment
source act-venv.sh

### Write which script to run below
python3 train_eval.py --conf_path $conf_path