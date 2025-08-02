#!/bin/bash

ACTIVATE="source ~/.bashrc && conda activate dmp"

temperature=0.8
reuse_prob=0.8
reweight_scaling=1.4
reweight_threshold=0.9

SESSION1="baseline"
SESSION2="soft"
SESSION3="hard"
SESSION4="reweight"

# Define arrays of models and datasets to loop through
models=("llama2-7b-chat" "llama2-13b-chat" "mistral-7b-instruct" "falcon-7b-instruct")
datasets=("triviaqa" "nq_open" "SQuAD" "halueval")

# Create 4 tmux sessions, one for each GPU
# Each session will run all experiments sequentially

# Session 1: Baseline experiments on GPU 3
baseline_jobs=""
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        session_name="${SESSION1}_${model}_${dataset}"
        job="$ACTIVATE && CUDA_VISIBLE_DEVICES=3 python -m pipeline.generate --model $model --dataset $dataset \
         --num_samples 400 --project_ind 0 --prompt_mode sentence --experiment_lot $session_name \
         --temperature $temperature"
        baseline_jobs="$baseline_jobs && echo 'Starting baseline: $model-$dataset' && $job && echo 'Completed baseline: $model-$dataset'"
    done
done

# Session 2: Soft reuse experiments on GPU 2
soft_jobs=""
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        session_name="${SESSION2}_${model}_${dataset}"
        job="$ACTIVATE && CUDA_VISIBLE_DEVICES=2 python -m pipeline.generate --model $model --dataset $dataset \
         --num_samples 400 --project_ind 0 --prompt_mode sentence --experiment_lot $session_name \
         --enable_reuse --temperature $temperature --reuse_mode soft"
        soft_jobs="$soft_jobs && echo 'Starting soft: $model-$dataset' && $job && echo 'Completed soft: $model-$dataset'"
    done
done

# Session 3: Hard reuse experiments on GPU 1
hard_jobs=""
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        session_name="${SESSION3}_${model}_${dataset}"
        job="$ACTIVATE && CUDA_VISIBLE_DEVICES=1 python -m pipeline.generate --model $model --dataset $dataset \
         --num_samples 400 --project_ind 0 --prompt_mode sentence --experiment_lot $session_name \
         --enable_reuse --temperature $temperature --reuse_mode hard --reuse_prob $reuse_prob"
        hard_jobs="$hard_jobs && echo 'Starting hard: $model-$dataset' && $job && echo 'Completed hard: $model-$dataset'"
    done
done

# Session 4: Reweight experiments on GPU 0
reweight_jobs=""
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        session_name="${SESSION4}_${model}_${dataset}"
        job="$ACTIVATE && CUDA_VISIBLE_DEVICES=0 python -m pipeline.generate --model $model --dataset $dataset \
         --num_samples 400 --project_ind 0 --prompt_mode sentence --experiment_lot $session_name \
         --enable_reuse --temperature $temperature --reuse_mode hard --reuse_prob $reuse_prob \
         --enable_reweight --reweight_scaling $reweight_scaling --reweight_threshold $reweight_threshold"
        reweight_jobs="$reweight_jobs && echo 'Starting reweight: $model-$dataset' && $job && echo 'Completed reweight: $model-$dataset'"
    done
done

# Create the 4 tmux sessions
echo "Creating tmux session for baseline experiments (GPU 3)..."
tmux new-session -d -s baseline_experiments "echo 'Starting baseline experiments...' $baseline_jobs && echo 'All baseline experiments completed!'"

echo "Creating tmux session for soft reuse experiments (GPU 2)..."
tmux new-session -d -s soft_experiments "echo 'Starting soft reuse experiments...' $soft_jobs && echo 'All soft reuse experiments completed!'"

echo "Creating tmux session for hard reuse experiments (GPU 1)..."
tmux new-session -d -s hard_experiments "echo 'Starting hard reuse experiments...' $hard_jobs && echo 'All hard reuse experiments completed!'"

echo "Creating tmux session for reweight experiments (GPU 0)..."
tmux new-session -d -s reweight_experiments "echo 'Starting reweight experiments...' $reweight_jobs && echo 'All reweight experiments completed!'"

echo "All 4 tmux sessions created successfully!"
echo "You can monitor them with:"
echo "  tmux attach -t baseline_experiments"
echo "  tmux attach -t soft_experiments"
echo "  tmux attach -t hard_experiments"
echo "  tmux attach -t reweight_experiments"

