#!/bin/bash

# submits all bsub jobs to be run

export data_dir=data/SP500_stock_prices_log_clean_3monthreturn_sentiment.csv
export exp_dir=experiments_sentiment

for script in training_shell_scripts/*; do
        echo $script
        if [[ -d $script ]]; then
            echo "skipping directory"
        fi
            envsubst < $script | bsub
done