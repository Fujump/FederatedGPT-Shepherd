import os
from typing import List
from tqdm import tqdm
import fire
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient
import datasets
from utils.prompter import Prompter

datasets.utils.logging.set_verbosity_error()


def fl_finetune(
        # model/data params
        global_model: str = '',
        data_path: str = './data',
        output_dir: str = './lora-shepherd/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.1,
        num_communication_rounds: int = 50,
        num_clients: int = 10,
        # Local training hyperparams
        local_batch_size: int = 64,  # 64,
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 10,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    data_path = os.path.join(data_path, str(num_clients))
    assert (os.path.exists(data_path), "Please generate the data files for each client")

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        global_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(global_model)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       group_by_length,
                                       ddp)

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client

        print("Collecting the weights of clients and performing aggregation")
        
        # # ...existing code...
        # print("Collecting the weights of clients and performing aggregation")

        # # 统计并记录权重的大小和方向
        # for name, module in model.named_modules():
        #     if hasattr(module, "weight"):
        #         weight = module.weight.data
        #         magnitude = torch.linalg.norm(weight, dim=1, keepdim=True)
        #         direction = weight / magnitude
        #         # 记录到文件
        #         torch.save(
        #             {"magnitude": magnitude.cpu(), "direction": direction.cpu()},
        #             os.path.join(output_dir, str(epoch), f"{name}_weight_stats.pt")
        #         )
        # # ...existing code...


        weight_stats_dir = os.path.join(output_dir, "weight_stats")
        os.makedirs(weight_stats_dir, exist_ok=True)
        # 假设初始权重统计已保存为 initial_weight_stats.pt
        if epoch == 0:
            initial_weight_stats = {}
            for name, module in model.named_modules():
                # if any(x in name for x in ["q_proj", "k_proj", "v_proj", "lora_up", "lora_down"]):
                if "lora_A.default" in name or "lora_B.default" in name:
                    if hasattr(module, 'weight'):
                        weight = module.weight.data
                        magnitude = torch.linalg.norm(weight.float(), dim=1, keepdim=True)
                        direction = weight / magnitude
                        initial_weight_stats[name] = {
                            "magnitude": magnitude.cpu(),
                            "direction": direction.cpu()
                        }
            torch.save(initial_weight_stats, os.path.join(weight_stats_dir, "initial_weight_stats.pt"))

        initial_stats = torch.load(os.path.join(weight_stats_dir, "initial_weight_stats.pt"))
        
        # 统计并记录权重的大小和方向，以及 delta_M^t 和 delta_D^t
        for name, module in model.named_modules():
            # if any(x in name for x in ["q_proj", "k_proj", "v_proj", "lora_up", "lora_down"]):
            if "lora_A.default" in name or "lora_B.default" in name:
                if hasattr(module, 'weight'):
                    weight = module.weight.data
                    magnitude = torch.linalg.norm(weight.float(), dim=1, keepdim=True)
                    direction = weight / magnitude
                    
                    # 加载初始权重统计
                    initial_magnitude = initial_stats[name]["magnitude"].to(weight.device)
                    initial_direction = initial_stats[name]["direction"].to(weight.device)
                    
                    # 计算 delta_M^t = Σ|m^{n,t} - m_0^n| / k
                    k = magnitude.shape[0]  # k = # out_features
                    delta_M = torch.abs(magnitude - initial_magnitude).sum() / k
                    
                    # 计算 delta_D^t = Σ(1 - cos(V^{n,t}, W_0^n)) / k
                    cos_sim = torch.sum(direction * initial_direction, dim=1)
                    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # 确保数值稳定性
                    delta_D = (1 - cos_sim).sum() / k
                    
                    # 记录到文件
                    os.makedirs(os.path.join(weight_stats_dir, str(epoch)), exist_ok=True)
                    torch.save(
                        {
                            "magnitude": magnitude.cpu(),
                            "direction": direction.cpu(),
                            "delta_M": delta_M.cpu(),
                            "delta_D": delta_D.cpu()
                        },
                        os.path.join(weight_stats_dir, str(epoch), f"{name}_weight_stats.pt")
                    )
        
        model = FedAvg(model,
                       selected_clients_set,
                       output_dir,
                       local_dataset_len_dict,
                       epoch,
                       )
        torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        config.save_pretrained(output_dir)

        # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
        global_evaluation()


if __name__ == "__main__":
    fire.Fire(fl_finetune)
