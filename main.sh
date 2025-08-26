python3 main.py --global_model 'chavinlo/alpaca-native'\
      --data_path  "./data" \
      --output_dir  './lora-shepherd-7b/'\
      --num_communication_rounds 10 \
      --num_clients  1 \
      --train_on_inputs \
      --group_by_length