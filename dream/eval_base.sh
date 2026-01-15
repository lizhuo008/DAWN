# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# model="/workplace/models/Dream/Dream-v0-Instruct-7B"
# model_name="Dream-v0-Instruct-7B"

model="/workplace/models/Dream/Dream-v0-Base-7B"
model_name="Dream-v0-Base-7B"


device=1

############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))


# # baseline
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/baseline/${task}-ns0-${length}


# # parallel
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/parallel/${task}-ns0-${length}

# # g-dllm
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=g-dllm,show_speed=True,conf_threshold=0.7,outp_path=evals_results_${model_name}/g-dllm/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/g-dllm/${task}-ns0-${length}

# # klass
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=klass,show_speed=True,conf_threshold=0.7,kl_threshold=0.015,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/klass/${task}-ns0-${length}

# # local leap
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=local_leap,conf_threshold=0.9,relaxed_threshold=0.8,radius=4,show_speed=True,outp_path=evals_results_${model_name}/local_leap/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/local_leap/${task}-ns0-${length} --log_samples

############################################### minerva_math evaluations ###############################################
task=minerva_math
length=256
block_length=32
num_fewshot=4
steps=$((length / block_length))

# # baseline
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/baseline/${task}-ns0-${length}


# # parallel
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/parallel/${task}-ns0-${length}

# # g-dllm
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=g-dllm,show_speed=True,conf_threshold=0.7,outp_path=evals_results_${model_name}/g-dllm/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/g-dllm/${task}-ns0-${length}

# # klass
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=klass,show_speed=True,conf_threshold=0.7,kl_threshold=0.015,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/klass/${task}-ns0-${length}

# # local leap
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=local_leap,conf_threshold=0.9,relaxed_threshold=0.8,radius=4,show_speed=True,outp_path=evals_results_${model_name}/local_leap/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/local_leap/${task}-ns0-${length} --log_samples
    
############################################### humaneval evaluations ###############################################
task=humaneval
length=256
block_length=32
num_fewshot=0
steps=$((length / block_length))

# # baseline
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/baseline/${task}-ns0-${length}


# # parallel
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/parallel/${task}-ns0-${length}

# # g-dllm
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=g-dllm,show_speed=True,conf_threshold=0.7,outp_path=evals_results_${model_name}/g-dllm/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/g-dllm/${task}-ns0-${length}

# # klass
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=klass,show_speed=True,conf_threshold=0.7,kl_threshold=0.015,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/klass/${task}-ns0-${length}

# # local leap
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=local_leap,conf_threshold=0.9,relaxed_threshold=0.8,radius=4,show_speed=True,outp_path=evals_results_${model_name}/local_leap/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/local_leap/${task}-ns0-${length} --log_samples

############################################### mbpp evaluations ###############################################
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=$((length / block_length))

# # baseline
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/baseline/${task}-ns0-${length}


# # parallel
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/parallel/${task}-ns0-${length}

# # g-dllm
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=g-dllm,show_speed=True,conf_threshold=0.7,outp_path=evals_results_${model_name}/g-dllm/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/g-dllm/${task}-ns0-${length}

# # klass
# CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=klass,show_speed=True,conf_threshold=0.7,kl_threshold=0.015,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --confirm_run_unsafe_code \
#     --output_path evals_results_${model_name}/klass/${task}-ns0-${length}

# # local leap
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=local_leap,conf_threshold=0.9,relaxed_threshold=0.8,radius=4,show_speed=True,outp_path=evals_results_${model_name}/local_leap/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/local_leap/${task}-ns0-${length} --log_samples