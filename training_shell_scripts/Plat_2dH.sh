#!/bin/sh 
pwd
#BSUB -q hpc
#BSUB -J "P_2dH"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 24:00
#BSUB -u simondanielschneider@gmail.com
#BSUB -N

### Doesnt overwrite, just makes new.
#BSUB -o outputs/plat_2dH_Output_%J.out
#BSUB -e errors/plat_2dH_Error_%J.err

### Activate virtual environment
source act-venv.sh

### Write which script to run below
python3 train_eval.py --conf_path configs/Plat/conf_2dH.yml --data_dir ${data_dir} --exp_dir ${exp_dir}
# --data_dir data/SP500_stock_prices_log_clean_3monthreturn_sentiment.csv --exp_dir experiments_sentiment