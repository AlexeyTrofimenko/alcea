#!/bin/bash
source ./miniconda/etc/profile.d/conda.sh

conda deactivate 2>/dev/null

if [ "$1" = "book" ]; then
    conda activate ./envs/booknlp_env
elif [ "$1" = "llm" ]; then
    conda activate ./envs/llm_env
elif [ "$1" = "flask" ]; then
    conda activate ./envs/flask_env
else
    echo "Usage: source env.sh [book|llm|flask]"
fi