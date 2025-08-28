# Copilot Instructions for FederatedGPT-Shepherd

## Project Overview
- **FederatedGPT-Shepherd** is a platform for federated instruction tuning of LLMs, enabling privacy-preserving, distributed fine-tuning across multiple clients.
- Main workflow: data preparation → federated fine-tuning → aggregation → evaluation → inference.
- Core technologies: PyTorch, HuggingFace Transformers, PEFT/LoRA, bitsandbytes, Gradio UI.

## Architecture & Key Components
- **main.py**: Entry point for federated training. Orchestrates client selection, local training, aggregation, and statistics logging.
- **fed_utils/**: Contains modules for client logic (`client.py`), aggregation (`model_aggregation.py`), scheduling, and evaluation.
- **utils/**: Helper modules for prompt templates (`prompter.py`) and streaming output (`callbacks.py`).
- **GlobalModel_generated.py**: Loads the global model and LoRA weights for inference via Gradio UI.
- **client_data_allocation.py**: Prepares per-client datasets from a central corpus.

## Developer Workflow
- **Install dependencies**: `pip install -r requirements.txt` (see README for bitsandbytes notes).
- **Prepare data**: `python client_data_allocation.py <num_clients> <diff_quantity>`
- **Run federated training**: `python main.py --global_model <model> --data_path ./data --output_dir ./lora-shepherd-7b/ ...`
- **Aggregate models**: Uses weighted averaging in `FedAvg` (see `fed_utils/model_aggregation.py`).
- **Inference**: `python GlobalModel_generated.py --base_model <model> --lora_weights_path <path> --lora_config_path <path>`

## Patterns & Conventions
- **Parameter-efficient tuning**: LoRA adapters are used for local updates; only adapter weights are aggregated.
- **Weight statistics**: During training, magnitude and direction of weights are logged for analysis (see `main.py`).
- **Client logic**: Each client is a `GeneralClient` instance, handling its own data, trainer, and local model updates.
- **Aggregation**: Only selected clients' weights are averaged, weighted by local dataset size.
- **Prompting**: Prompter class manages prompt templates for instruction tuning.
- **Evaluation**: Customizable via `fed_utils/evaluation.py`.

## Integration Points
- **External models**: Supports LLaMA, Alpaca, Vicuna, Baize, and more via HuggingFace.
- **PEFT/LoRA**: Adapter logic is central; see `main.py` and `fed_utils/client.py` for usage.
- **bitsandbytes**: For 8-bit model loading and training efficiency.
- **Gradio**: Used for interactive inference UI.

## Examples
- **Federated training command**:
  ```bash
  python main.py --global_model 'chavinlo/alpaca-native' --data_path ./data --output_dir ./lora-shepherd-7b/ --num_communication_rounds 10 --num_clients 10 --train_on_inputs --group_by_length
  ```
- **Inference command**:
  ```bash
  python GlobalModel_generated.py --base_model 'chavinlo/alpaca-native' --lora_weights_path /output/path/to/lora_weights --lora_config_path /output/path/to/lora_config
  ```

## Key Files & Directories
- `main.py`, `fed_utils/`, `utils/`, `GlobalModel_generated.py`, `client_data_allocation.py`, `requirements.txt`, `README.md`

---
If any section is unclear or missing, please provide feedback for further refinement.
