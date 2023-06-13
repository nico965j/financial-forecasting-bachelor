#!/bin/bash

# submits all bsub jobs to be run

for script in training_shell_scripts/*; do
        echo $script
        if [[ -d $script ]]; then
            echo "skipping directory"
        fi
            bsub < $script

done
