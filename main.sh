python3 main.py --global_model 'chavinlo/alpaca-native'\
      --data_path  "./data_mini" \
      --output_dir  './lora-shepherd-7b_mini/'\
      --num_communication_rounds 5 \
      --num_clients  1 \
      --train_on_inputs \
      --group_by_length