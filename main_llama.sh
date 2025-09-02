python3 main.py --global_model 'decapoda-research/llama-7b-hf'\
      --data_path  "./data_1" \
      --output_dir  './lora-shepherd-7b_llama/'\
      --num_communication_rounds 1 \
      --num_clients  1 \
      --local_num_epochs 1 \
      --train_on_inputs \
      --group_by_length