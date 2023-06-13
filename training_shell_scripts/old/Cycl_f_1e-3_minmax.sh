#!/bin/sh 
pwd
#BSUB -q hpc
#BSUB -J "Cycl"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 24:00
#BSUB -u simondanielschneider@gmail.com
#BSUB -N

### Doesnt overwrite, just makes new.
#BSUB -o outputs/Cycl_Output_%J.out
#BSUB -e errors/Cycl_Error_%J.err

conf_path=configs/Cycl_f_1e-3_minmax.yml

### Activate virtual environment
source act-venv.sh

### Write which script to run below
python3 train_eval.py --conf_path $conf_path