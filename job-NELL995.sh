
params_tasks=("2p" "3p" "2i" "3i" "ip" "pi" "up")
cqd_k=""
cqd_t_norm=""
for param_task in "${params_tasks[@]}"; do
    if [ "$param_task" = "2p" ]; then
      cqd_k="512"
      cqd_t_norm="prod"
      params_subtasks=("None" "1p" "2p") # For the new benchmarks use "New"
    elif [ "$param_task" = "3p" ]; then
      cqd_k="8"
      cqd_t_norm="prod"
      params_subtasks=("None" "1p" "2p" "3p")
      echo "params_subtasks: ${params_subtasks[@]}"
    elif [ "$param_task" = "2i" ]; then
      cqd_k="128"
      cqd_t_norm="prod"
      params_subtasks=("None")
    elif [ "$param_task" = "3i" ]; then
      cqd_k="128"
      cqd_t_norm="prod"
      params_subtasks=("None" "1p" "2i" "3i")
    elif [ "$param_task" = "ip" ]; then
      cqd_k="64"
      cqd_t_norm="prod"
      params_subtasks=("None" "1p" "2p" "2i" "ip")
    elif [ "$param_task" = "pi" ]; then
      cqd_k="256"
      cqd_t_norm="prod"
      params_subtasks=("None" "1p" "2p" "2i" "pi")
    elif [ "$param_task" = "2u" ]; then
      cqd_k="512"
      cqd_t_norm="min"
      params_subtasks=("None" "filt")
    elif [ "$param_task" = "up" ]; then
      cqd_k="512"
      cqd_t_norm="min"
      params_subtasks=("None" "filt" "1p" "2u" "up")
    fi
    for params_subtask in "${params_subtasks[@]}"; do
        python main.py --do_test --data_path NELL995 --new_bench_path NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --print_on_screen --test_batch_size 1 --checkpoint_path lp_models/icews18 --cqd discrete --cqd-t-norm "$cqd_t_norm" --cqd-k $cqd_k --tasks "$param_task" --subtask "$params_subtask" --cqd-max-norm 0.9 --cqd-normalize --cqd-max-k 512
    done
done