#!/bin/sh 
pwd
#BSUB -q hpc
#BSUB -J "C_2dh"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 24:00
#BSUB -u simondanielschneider@gmail.com
#BSUB -N

### Doesnt overwrite, just makes new.
#BSUB -o outputs/cycl_2dh_Output_%J.out
#BSUB -e errors/cycl_2dh_Error_%J.err

### Activate virtual environment
source act-venv.sh

### Write which script to run below
python3 train_eval.py --conf_path configs/Cycl/conf_2dh.yml