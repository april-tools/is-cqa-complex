#source hyp_kg_env/bin/activate

params_tasks=("2p" "3p" "2i" "3i" "ip" "pi" "up")
cqd_k=""
cqd_t_norm=""
for param_task in "${params_tasks[@]}"; do
    if [ "$param_task" = "2p" ]; then
      cqd_k="512"
      cqd_t_norm="prod"
      params_subtasks=("1p" "2p" "None")
    elif [ "$param_task" = "3p" ]; then
      cqd_k="8"
      cqd_t_norm="prod"
      params_subtasks=("1p" "2p" "3p" "None")
      echo "params_subtasks: ${params_subtasks[@]}"
    elif [ "$param_task" = "2i" ]; then
      cqd_k="128"
      cqd_t_norm="prod"
      params_subtasks=("1p" "2i" "None")
    elif [ "$param_task" = "3i" ]; then
      cqd_k="128"
      cqd_t_norm="prod"
      params_subtasks=("1p" "2i" "3i" "None")
    elif [ "$param_task" = "ip" ]; then
      cqd_k="64"
      cqd_t_norm="prod"
      params_subtasks=("1p" "2p" "2i" "ip" "None")
    elif [ "$param_task" = "pi" ]; then
      cqd_k="256"
      cqd_t_norm="prod"
      params_subtasks=("1p" "2p" "2i" "ip" "None")
    elif [ "$param_task" = "up" ]; then
      cqd_k="512"
      cqd_t_norm="min"
      params_subtasks=("1p" "2p" "2u" "up" "None")
    fi
    for params_subtask in "${params_subtasks[@]}"; do
        python main.py --do_test --data_path data/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --print_on_screen --test_batch_size 1 --checkpoint_path models/fb15k-237-betae --cqd discrete --cqd-t-norm "$cqd_t_norm" --cqd-k $cqd_k --tasks "$param_task" --subtask "$params_subtask" --cqd-normalize --cqd-max-k 16
    done
done