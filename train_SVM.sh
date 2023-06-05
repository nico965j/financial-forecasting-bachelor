#!/bin/sh 
#BSUB -q hpc
#BSUB -J "SVM"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 24:00 
#BSUB -u simondanielschneider@gmail.com
#BSUB -N

### Doesnt overwrite, just makes new.
#BSUB -o outputs/SVM_Output_%J.out
#BSUB -e errors/SVM_Error_%J.err

### Activate virtual environment
source act-venv.sh

### Write which script to run below
python3 SVM_server.py