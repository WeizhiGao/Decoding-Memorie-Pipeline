#!/bin/bash

ACTIVATE="source ~/.bashrc && conda activate dmp"

temperature=0.8
reuse_prob=0.8
reweight_scaling=1.1
reweight_threshold=0.9

SESSION1="baseline"
SESSION2="soft"
SESSION3="hard"
SESSION4="reweight"

models=("Llama2-7b-chat" "Llama2-13b-chat" "mistral-7b-instruct" "falcon-7b-instruct")
datasets=("trivia_qa" "nq" "squad" "halueval")

baseline_jobs=""
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        session_name="${SESSION1}_${model}_${dataset}"
        job="$ACTIVATE && CUDA_VISIBLE_DEVICES=3 python generate_answers.py --model_name=$model --dataset=$dataset \
         --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable --no-compute_p_true \
         --no-compute_accuracy_at_all_temps --no-compute_context_entails_response --no-compute_uncertainties \
         --brief_prompt=chat --model_max_new_tokens=100 --num_few_shot=0 --metric=llm_gpt-4 --entailment_model=gpt-3.5 \
         --num_samples=400 --num_generations=10 --temperature $temperature --topk 10 --topp 0.9 --experiment_lot $session_name"
        baseline_jobs="$baseline_jobs && echo 'Starting baseline: $model-$dataset' && $job && echo 'Completed baseline: $model-$dataset'"
    done
done

soft_jobs=""
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        session_name="${SESSION2}_${model}_${dataset}"
        job="$ACTIVATE && CUDA_VISIBLE_DEVICES=2 python generate_answers.py --model_name=$model --dataset=$dataset \
         --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable --no-compute_p_true \
         --no-compute_accuracy_at_all_temps --no-compute_context_entails_response --no-compute_uncertainties \
         --brief_prompt=chat --model_max_new_tokens=100 --num_few_shot=0 --metric=llm_gpt-4 --entailment_model=gpt-3.5 \
         --num_samples=400 --num_generations=10 --temperature $temperature --topk 10 --topp 0.9 --experiment_lot $session_name \
         --enable_reuse --reuse_mode soft"
        soft_jobs="$soft_jobs && echo 'Starting soft: $model-$dataset' && $job && echo 'Completed soft: $model-$dataset'"
    done
done

hard_jobs=""
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        session_name="${SESSION3}_${model}_${dataset}"
        job="$ACTIVATE && CUDA_VISIBLE_DEVICES=1 python generate_answers.py --model_name=$model --dataset=$dataset \
         --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable --no-compute_p_true \
         --no-compute_accuracy_at_all_temps --no-compute_context_entails_response --no-compute_uncertainties \
         --brief_prompt=chat --model_max_new_tokens=100 --num_few_shot=0 --metric=llm_gpt-4 --entailment_model=gpt-3.5 \
         --num_samples=400 --num_generations=10 --temperature $temperature --topk 10 --topp 0.9 --experiment_lot $session_name \
         --enable_reuse --reuse_mode hard --reuse_prob $reuse_prob"
        hard_jobs="$hard_jobs && echo 'Starting hard: $model-$dataset' && $job && echo 'Completed hard: $model-$dataset'"
    done
done

reweight_jobs=""
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        session_name="${SESSION4}_${model}_${dataset}"
        job="$ACTIVATE && CUDA_VISIBLE_DEVICES=0 python generate_answers.py --model_name=$model --dataset=$dataset \
         --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable --no-compute_p_true \
         --no-compute_accuracy_at_all_temps --no-compute_context_entails_response --no-compute_uncertainties \
         --brief_prompt=chat --model_max_new_tokens=100 --num_few_shot=0 --metric=llm_gpt-4 --entailment_model=gpt-3.5 \
         --num_samples=400 --num_generations=10 --temperature $temperature --topk 10 --topp 0.9 --experiment_lot $session_name \
         --enable_reuse --reuse_mode hard --reuse_prob $reuse_prob \
         --enable_reweight  --reweight_scaling $reweight_scaling --threshold $reweight_threshold"
        reweight_jobs="$reweight_jobs && echo 'Starting reweight: $model-$dataset' && $job && echo 'Completed reweight: $model-$dataset'"
    done
done

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
