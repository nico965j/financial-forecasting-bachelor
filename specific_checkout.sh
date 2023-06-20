#!/bin/bash

while IFS= read -r file
do
    git checkout origin/simon -- "$file"
done < files_to_pull_simon.txt
