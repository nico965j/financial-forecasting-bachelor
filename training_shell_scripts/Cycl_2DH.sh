#!/bin/sh 
pwd
#BSUB -q hpc
#BSUB -J "C_2DH"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 24:00
#BSUB -u simondanielschneider@gmail.com
#BSUB -N

### Doesnt overwrite, just makes new.
#BSUB -o outputs/cycl_2DH_Output_%J.out
#BSUB -e errors/cycl_2DH_Error_%J.err

### Activate virtual environment
source act-venv.sh

### Write which script to run below
python3 train_eval.py --conf_path configs/Cycl/conf_2DH.yml --data_dir ${data_dir} --exp_dir ${exp_dir}