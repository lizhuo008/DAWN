# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model_path='GSAI-ML/LLaDA-8B-Instruct'
model_name='LLaDA-8B-Instruct'

device=0

############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/baseline/${task}-ns0-${length}

# parallel threshold
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/parallel/${task}-ns0-${length}

# klass
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=64,show_speed=True,klass=True,threshold_klass=0.6,kl_threshold=0.015,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/klass/${task}-ns0-${length}

# local leap
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,local_leap=True,threshold=0.9,relaxed_threshold=0.75,radius=4,outp_path=evals_results_${model_name}/local_leap/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/local_leap/${task}-ns0-${length}

# dawn
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dawn=True,tau_sink=0.01,tau_edge=0.07,tau_induce=0.7,tau_low=0.75,outp_path=evals_results_${model_name}/dawn/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dawn/${task}-ns0-${length}

############################################### minerva_math evaluations ###############################################
task=minerva_math
length=256
block_length=32
num_fewshot=4
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/baseline/${task}-ns0-${length}

# parallel threshold
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/parallel/${task}-ns0-${length}

# klass
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 1 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=64,show_speed=True,klass=True,threshold_klass=0.6,kl_threshold=0.01,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/klass/${task}-ns0-${length}

#local leap
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,local_leap=True,threshold=0.9,relaxed_threshold=0.75,radius=4,outp_path=evals_results_${model_name}/local_leap/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/local_leap/${task}-ns0-${length}

# dawn
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dawn=True,tau_sink=0.01,tau_edge=0.07,tau_induce=0.7,tau_low=0.75,outp_path=evals_results_${model_name}/dawn/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dawn/${task}-ns0-${length}

############################################### gsm8k evaluations ###############################################
task=humaneval
length=256
block_length=32
num_fewshot=0
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples

# parallel threshold
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/parallel/${task}-ns0-${length} --log_samples

# klass
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 1 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=64,show_speed=True,klass=True,threshold_klass=0.9,kl_threshold=0.01,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/klass/${task}-ns0-${length} --log_samples

#local leap
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,local_leap=True,threshold=0.9,relaxed_threshold=0.75,radius=4,outp_path=evals_results_${model_name}/local_leap/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/local_leap/${task}-ns0-${length} --log_samples

# dawn
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dawn=True,tau_sink=0.01,tau_edge=0.07,tau_induce=0.7,tau_low=0.8,outp_path=evals_results_${model_name}/dawn/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dawn/${task}-ns0-${length} --log_samples

## NOTICE: use postprocess for humaneval
# python3 postprocess_code_humaneval.py {the samples_xxx.jsonl file under output_path}

############################################### mbpp evaluations ###############################################
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples

# parallel threshold
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/parallel/${task}-ns0-${length} --log_samples

# klass
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 1 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=64,show_speed=True,klass=True,threshold_klass=0.7,kl_threshold=0.01,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/klass/${task}-ns0-${length} --log_samples

#local leap
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,local_leap=True,threshold=0.9,relaxed_threshold=0.75,radius=4,outp_path=evals_results_${model_name}/local_leap/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/local_leap/${task}-ns0-${length} --log_samples

# dawn
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,dawn=True,tau_sink=0.01,tau_edge=0.07,tau_induce=0.7,tau_low=0.7,outp_path=evals_results_${model_name}/dawn/${task}-ns0-${length}/results.jsonl \
--output_path evals_results_${model_name}/dawn/${task}-ns0-${length} --log_samples

## NOTICE: use postprocess for mbpp
# python postprocess_code_mbpp.py {the samples_xxx.jsonl file under output_path}