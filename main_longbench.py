import os
import sys
import json
import torch
import argparse
import numpy as np
import logging
from transformers import AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
from importlib.metadata import version

from lib.prune import prune_flap, prune_wanda_sp, prune_magnitude_sp, calculate_model_params, calculate_query_params, calculate_key_params, calculate_value_params, calculate_output_params, analyze_linear_layers

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

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

def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def evaluate_longbench(args, model, tokenizer, device):
    # Import necessary metrics
    from metrics import (
        qa_f1_score, rouge_zh_score, qa_f1_zh_score, rouge_score,
        classification_score, retrieval_score, retrieval_zh_score,
        count_score, code_sim_score
    )
    
    # Define dataset to metric mapping
    dataset2metric = {
        "narrativeqa": qa_f1_score,
        "qasper": qa_f1_score,
        "multifieldqa_en": qa_f1_score,
        "hotpotqa": qa_f1_score,
        "2wikimqa": qa_f1_score,
        "musique": qa_f1_score,
        "gov_report": rouge_score,
        "qmsum": rouge_score,
        "multi_news": rouge_score,
        "trec": classification_score,
        "triviaqa": qa_f1_score,
        "samsum": rouge_score,
        "passage_count": count_score,
        "passage_retrieval_en": retrieval_score,
        "lcc": code_sim_score,
        "repobench-p": code_sim_score,
    }
    
    # Load configuration for LongBench
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    # Only evaluate datasets from the LaTeX table
    # These match the columns in the table
    datasets = [
        "narrativeqa",          # NrtvQA
        "qasper",               # Qasper
        "multifieldqa_en",      # MF-en
        "hotpotqa",             # HotpotQA
        "2wikimqa",             # 2WikiMQA
        "musique",              # Musique
        "gov_report",           # GovReport
        "qmsum",                # QMSum
        "multi_news",           # MultiNews
        "trec",                 # TREC
        "triviaqa",             # TriviaQA
        "samsum",               # SAMSum
        "passage_count",        # PCount
        "passage_retrieval_en", # PRe
        "lcc",                  # Lcc
        "repobench-p"           # RB-P
    ]

    # Create result structure
    results = {}
    
    # Helper function for scoring
    def scorer(dataset, predictions, answers, all_classes):
        total_score = 0.
        for (prediction, ground_truths) in zip(predictions, answers):
            score = 0.
            if dataset in ["trec", "triviaqa", "samsum"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
            total_score += score
        return round(100 * total_score / len(predictions), 2)
    
    # Evaluate on each dataset
    for dataset in datasets:
        print(f"Evaluating on {dataset}...")
        
        # Load dataset
        data = load_dataset('THUDM/LongBench', dataset, split='test')
            
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        predictions = []
        answers = []
        all_classes = []
        
        # Process each sample
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            
            # Store answer info
            answers.append(json_obj["answers"])
            if "all_classes" in json_obj:
                all_classes = json_obj["all_classes"]
            
            # Truncate to fit max_length
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if "chatglm3" in args.prune_model:
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
                
            if len(tokenized_prompt) > args.max_length:
                half = int(args.max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                
            # Apply chat formatting if needed
            if dataset not in ["trec", "triviaqa", "samsum", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt, args.prune_model)
                
            # Prepare input for generation
            if "chatglm3" in args.prune_model:
                if dataset in ["trec", "triviaqa", "samsum", "lcc", "repobench-p"]:
                    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                else:
                    input = prompt.to(device)
            else:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                
            context_length = input.input_ids.shape[-1] if hasattr(input, 'input_ids') else 0
            
            # Generate output
            if dataset == "samsum":
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
                
            # Process output
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, args.prune_model)
            predictions.append(pred)
        
        # Calculate scores for this dataset
        results[dataset] = scorer(dataset, predictions, answers, all_classes)
        print(f"Score for {dataset}: {results[dataset]}")
        
    return results

def main():
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
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length for evaluation")
    parser.add_argument("--verbosity", default="INFO", help="Logging level: CRITICAL, ERROR, WARNING, INFO, DEBUG.")
    parser.add_argument("--show_config", action="store_true", default=False, help="If True, shows the full config of all tasks at the end of the evaluation.")
    parser.add_argument("--output_path", type=str, default=None, help="Path for saving results.")

    args = parser.parse_args()

    # Model-specific configurations
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
        print("Error: Unsupported model!")
        sys.exit(1)

    # Alpha lists for different models
    alpha_lists = {
        "mistralai/Mistral-7B-v0.3": [0.7, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0],
        "google/gemma-2-9b": [0.5, 0.4, 0.3, 0.4, 0.5, 0.4],
        "meta-llama/Meta-Llama-3.1-8B":  [0.8, 0.2, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0],
        "Qwen/Qwen2-7B": [0.9, 0.3, 0.3, 0.2],
        "meta-llama/Meta-Llama-3.1-70B": [],
        "Qwen/Qwen2-72B": []
    }

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Load model configuration
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
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(to_save_dir)
    device = torch.device("cuda")
    
    # Load model
    if args.quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_storage=torch.uint8)
        model = CustomCausalLM.from_pretrained(to_save_dir, torch_dtype=torch.bfloat16, config=config, quantization_config=bnb_config)
        print("Quantize: ", args.prune_model)
    else:
        model = CustomCausalLM.from_pretrained(to_save_dir, torch_dtype=torch.bfloat16, config=config)
        model.to(device)
        print("No Quantization: ", args.prune_model)

    # Calculate pre-pruning parameters
    before_prune_total_params = calculate_model_params(model)
    query_params = calculate_query_params(model)
    key_params = calculate_key_params(model)
    value_params = calculate_value_params(model)
    output_params = calculate_output_params(model)
    print(f"Before pruning, Total parameter {before_prune_total_params / 1000 ** 3:.8f}B; "
          f"Query parameter {query_params / 1000 ** 3:.8f}B; "
          f"Key parameter {key_params / 1000 ** 3:.8f}B")

    # Add bias if required
    if args.bias:
        for i in range(args.start_pruning_layer_idx, len(model.model.layers)):
            model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
                torch.zeros(model.model.layers[i].mlp.down_proj.weight.shape[0], 
                           device=model.model.layers[i].mlp.down_proj.weight.device, 
                           dtype=torch.bfloat16))
            torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)

    # Prepare model for pruning
    model.eval()
    model.seqlen = 128
    print("use device ", device)

    # Apply pruning method
    if args.prune_method == "flap":
        prune_flap(args, model, tokenizer, device)
    elif args.prune_method == "wanda_sp":
        prune_wanda_sp(args, model, tokenizer, device)
    elif args.prune_method == "mag_sp":
        prune_magnitude_sp(args, model, tokenizer, device)

    # Calculate post-pruning parameters
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

    # Initialize logger
    logger = setup_logging(args.verbosity)
    
    # Set model sequence length for LongBench evaluation
    model.seqlen = args.max_length
    
    # Run LongBench evaluation
    results = evaluate_longbench(args, model, tokenizer, device)
    
    # Save results
    model_short_name = args.prune_model.split('/')[-1]
    results_dir = os.path.join(args.result_dir, model_short_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct filename
    method_name = "harp+flap" if args.prune_method == "flap" else args.prune_method
    results_filename = f"{model_short_name}-{method_name}-longbench-table-{final_kv_cache_pruning_ratio}-{args.start_pruning_layer_idx}_{args.end_pruning_layer_idx}"
    results_path = os.path.join(results_dir, f"{results_filename}-results.json")
    
    # Add summary statistics to results
    avg_score = sum(results.values()) / len(results)
    results["_avg_score"] = round(avg_score, 2)
    print(f"Average score across all datasets: {avg_score:.2f}")
    
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2, default=_handle_non_serializable))
        
    if args.show_config or True:  # Always show results
        logger.info(json.dumps(results, indent=2, default=_handle_non_serializable))
    
    print(f"Results saved to {results_path}")
    
    # Print results in a format that can be directly copied to the LaTeX table
    print("\nResults for LaTeX table:")
    print(f"{results['narrativeqa']} & {results['qasper']} & {results['multifieldqa_en']} & "
          f"{results['hotpotqa']} & {results['2wikimqa']} & {results['musique']} & "
          f"{results['gov_report']} & {results['qmsum']} & {results['multi_news']} & "
          f"{results['trec']} & {results['triviaqa']} & {results['samsum']} & "
          f"{results['passage_count']} & {results['passage_retrieval_en']} & "
          f"{results['lcc']} & {results['repobench-p']} & {results['_avg_score']}")

if __name__ == '__main__':
    main()
