import os
import sys
import json
import torch
import argparse
import lm_eval
import numpy as np
import logging
from transformers import AutoTokenizer, BitsAndBytesConfig
from lm_eval.models.huggingface import HFLM
from lib.prune import prune_flap, prune_wanda_sp, prune_magnitude_sp, calculate_model_params, calculate_query_params, calculate_key_params, calculate_value_params, calculate_output_params, analyze_linear_layers

def setup_logging(verbosity):
    logging.basicConfig(
        level=verbosity.upper(), format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)

def load_task(task):
    return None, task.split(",") if task else []

def handle_output(args, results, logger):
    if not args.output_path:
        logger.info(json.dumps(results, indent=2, default=_handle_non_serializable))
        return

    results_str = json.dumps(results, indent=2, default=_handle_non_serializable)
    if args.show_config:
        logger.info(results_str)

    file_path = args.output_path + "-results.json"
    with open(file_path , "w", encoding="utf-8") as f:
        f.write(results_str)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=1024, help='Number of calibration samples.')
parser.add_argument('--pruning_ratio', type=float, default=0.125, help='Pruning ratio.')
parser.add_argument("--metrics", type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN", 'N/A'])
parser.add_argument("--prune_method", type=str, default="flap", choices=["flap", "wanda_sp", "mag_sp"])
parser.add_argument("--prune_model", type=str, choices=["mistralai/Mistral-7B-v0.3", 
                                                        "google/gemma-2-9b", 
                                                        "meta-llama/Meta-Llama-3-8B",
                                                        "Qwen/Qwen2-7B",
                                                        "meta-llama/Meta-Llama-3-70B",
                                                        "Qwen/Qwen2-72B",
                                                        "meta-llama/Meta-Llama-3.1-8B",
                                                        "meta-llama/Meta-Llama-3.1-70B"], default="mistralai/Mistral-7B-v0.3")
parser.add_argument("--model_dir", type=str, default="../model_dir")
parser.add_argument("--result_dir", type=str, default="result_dir")
parser.add_argument("--drop_last_layers", type=int, default=1)
parser.add_argument('--start_pruning_layer_idx', type=int, default=0, help='Layer idx post which pruning starts')
parser.add_argument('--end_pruning_layer_idx', type=int, default=32, help='Layer idx post which pruning starts')
parser.add_argument("--bias", action="store_true", default=False)
parser.add_argument("--quant", action="store_true", default=False)

parser.add_argument("--verbosity", default="INFO", help="Logging level: CRITICAL, ERROR, WARNING, INFO, DEBUG.")
parser.add_argument("--show_config", action="store_true", default=False, help="If True, shows the full config of all tasks at the end of the evaluation.")
parser.add_argument("--output_path", type=str, default=None, help="Path for saving results.")

args = parser.parse_args()

if args.prune_model.split("/")[0] == "mistralai":
    from models.configuration_mistral_drop_qk import MistralConfig as CustomConfig
    from models.modeling_mistral_drop_qk import MistralForCausalLM as CustomCausalLM
elif args.prune_model.split("/")[0] == "meta-llama":
    if "3.1" in args.prune_model.split("/")[1]:
        from models.configuration_llama3_1_drop_qk import LlamaConfig as CustomConfig
        from models.modeling_llama3_1_drop_qk import LlamaForCausalLM as CustomCausalLM
    else:
        from models.configuration_llama_drop_qk import LlamaConfig as CustomConfig
        from models.modeling_llama_drop_qk import LlamaForCausalLM as CustomCausalLM
elif args.prune_model.split("/")[0] == "google":
    from models.configuration_gemma2_drop_qk import Gemma2Config as CustomConfig
    from models.modeling_gemma2_drop_qk import Gemma2ForCausalLM as CustomCausalLM
elif args.prune_model.split("/")[0] == "Qwen":
    from models.configuration_qwen2_drop_qk import Qwen2Config as CustomConfig
    from models.modeling_qwen2_drop_qk import Qwen2ForCausalLM as CustomCausalLM
else:
    print("Error!!!")


alpha_lists = {
    "mistralai/Mistral-7B-v0.3": [0.7, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0],
    "google/gemma-2-9b": [0.5, 0.4, 0.3, 0.4, 0.5, 0.4],
    "meta-llama/Meta-Llama-3.1-8B":  [0.8, 0.2, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0],
    "Qwen/Qwen2-7B": [0.9, 0.3, 0.3, 0.2],
    "meta-llama/Meta-Llama-3.1-70B": [],
    "Qwen/Qwen2-72B": []
}

to_save_dir = os.path.join(args.model_dir, args.prune_model)
config_path = os.path.join(to_save_dir, "config.json")
with open(config_path, 'r') as f:
    config_dict = json.load(f)

if args.prune_model in alpha_lists:
    config_dict['alpha_list'] = alpha_lists[args.prune_model]
else:
    print(f"Error: Model {args.prune_model} does not have a predefined alpha list.")
    sys.exit(1)
config_dict['drop_qk_list'] = list(range(config_dict['num_hidden_layers'] - args.drop_last_layers, config_dict['num_hidden_layers']))
config = CustomConfig(**config_dict)
tokenizer = AutoTokenizer.from_pretrained(to_save_dir)
device = torch.device("cuda")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_storage=torch.uint8)
if not args.quant:
    model = CustomCausalLM.from_pretrained(to_save_dir, torch_dtype=torch.bfloat16, config=config)
    model.to(device)
    print("No Quantization: ", args.prune_model)
else:
    model = CustomCausalLM.from_pretrained(to_save_dir, torch_dtype=torch.bfloat16, config=config, quantization_config=bnb_config)
    print("Quantize: ", args.prune_model)

before_prune_total_params = calculate_model_params(model)
query_params = calculate_query_params(model)
key_params = calculate_key_params(model)
value_params = calculate_value_params(model)
output_params = calculate_output_params(model)
print(f"Before pruning, Total parameter {before_prune_total_params / 1000 ** 3:.8f}B; Query parameter {query_params / 1000 ** 3:.8f}B; Key parameter {key_params / 1000 ** 3:.8f}B")

if args.bias:
    for i in range(args.start_pruning_layer_idx, len(model.model.layers)):
        model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(torch.zeros(model.model.layers[i].mlp.down_proj.weight.shape[0], device=model.model.layers[i].mlp.down_proj.weight.device, dtype=torch.bfloat16))  # 或 'cuda'
        torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)

model.eval()
model.seqlen = 128
print("use device ", device)

# Prune the model
if args.prune_method == "flap":
    prune_flap(args, model, tokenizer, device)
elif args.prune_method == "wanda_sp":
    prune_wanda_sp(args, model, tokenizer, device)
elif args.prune_method == "mag_sp":
    prune_magnitude_sp(args, model, tokenizer, device)

after_prune_total_params = calculate_model_params(model)
prune_query_params = query_params / config_dict['num_hidden_layers'] * args.drop_last_layers
prune_key_params = key_params / config_dict['num_hidden_layers'] * args.drop_last_layers
prune_value_params = value_params / config_dict['num_hidden_layers'] * config_dict['alpha_list'][:args.drop_last_layers].count(0.0)
prune_output_params = output_params / config_dict['num_hidden_layers'] * config_dict['alpha_list'][:args.drop_last_layers].count(0.0)

after_prune_total_params = after_prune_total_params - prune_query_params - prune_key_params - prune_value_params - prune_output_params
print(f"After pruning, Total parameter {after_prune_total_params / 1000 ** 3:.8f}B")
print(f"Pruning ratio {1 - after_prune_total_params / before_prune_total_params:.8f}")
print(f"KV cache reduction ratio {args.drop_last_layers / config_dict['num_hidden_layers']:.8f}")
final_pruning_ratio = round(1 - after_prune_total_params / before_prune_total_params, 6)
final_kv_cache_pruning_ratio = round(args.drop_last_layers / config_dict['num_hidden_layers'], 4)
analyze_linear_layers(model)

task_num_fewshot_list = [
    ("nq_open", 5),
    ("winogrande", 5),
    ("arc_challenge", 25),
    ("boolq", 0),
    ("openbookqa", 0),
    ("piqa", 0),
    ("mmlu", 5),
    ("triviaqa", 5),
    ("gsm8k", 5),
    ("leaderboard_math_hard", 4),
    ("bbh_cot_fewshot", 3),
]

# Initialize logger and model
logger = setup_logging(args.verbosity)
lm = HFLM(pretrained=model, tokenizer=tokenizer, max_length=tokenizer.model_max_length)
for task, num_fewshot in task_num_fewshot_list:
    # Create result directory and output path
    os.makedirs(os.path.join(args.result_dir, args.prune_model), exist_ok=True)
    args.output_path = os.path.join(args.result_dir, args.prune_model, f"{args.prune_method}-{task}-{num_fewshot}-{args.start_pruning_layer_idx}_{args.end_pruning_layer_idx}-{final_kv_cache_pruning_ratio}-{final_pruning_ratio}")
    file_path = args.output_path + "-results.json"
    
    # Skip if results file already exists
    if os.path.exists(file_path):
        print(f"Output file {file_path} already exists. Skipping.")
        continue
    
    # Load task and evaluate
    task_manager, task_list = load_task(task)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=1,
        device="cuda",
        task_manager=task_manager
    )

    handle_output(args, results, logger)

