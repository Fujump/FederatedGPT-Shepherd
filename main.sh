python3 main.py --global_model 'chavinlo/alpaca-native'\
      --data_path  "./data_1" \
      --output_dir  './lora-shepherd-7b_1/'\
      --num_communication_rounds 1 \
      --num_clients  1 \
      --local_num_epochs 3 \
      --train_on_inputs \
      --group_by_length