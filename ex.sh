python3 main_longbench.py --prune_model meta-llama/Meta-Llama-3.1-8B --drop_last_layers 8 --prune_method flap --bias --start_pruning_layer_idx 0 --end_pruning_layer_idx 32 --pruning_ratio 0.01860
python3 main_longbench.py --prune_model mistralai/Mistral-7B-v0.3 --drop_last_layers 8 --prune_method flap --bias --start_pruning_layer_idx 0 --end_pruning_layer_idx 32 --pruning_ratio 0.01488
python3 main_longbench.py --prune_model Qwen/Qwen2-7B --drop_last_layers 4 --prune_method flap --bias --start_pruning_layer_idx 0 --end_pruning_layer_idx 28 --pruning_ratio 0.00923
python3 main_longbench.py --prune_model google/gemma-2-9b --drop_last_layers 6 --prune_method flap --bias --start_pruning_layer_idx 0 --end_pruning_layer_idx 42 --pruning_ratio 0.02042


python3 main.py --prune_model meta-llama/Meta-Llama-3.1-8B --drop_last_layers 8 --prune_method flap --bias --start_pruning_layer_idx 0 --end_pruning_layer_idx 32 --pruning_ratio 0.01860
python3 main.py --prune_model mistralai/Mistral-7B-v0.3 --drop_last_layers 8 --prune_method flap --bias --start_pruning_layer_idx 0 --end_pruning_layer_idx 32 --pruning_ratio 0.01488
python3 main.py --prune_model Qwen/Qwen2-7B --drop_last_layers 4 --prune_method flap --bias --start_pruning_layer_idx 0 --end_pruning_layer_idx 28 --pruning_ratio 0.00923
python3 main.py --prune_model google/gemma-2-9b --drop_last_layers 6 --prune_method flap --bias --start_pruning_layer_idx 0 --end_pruning_layer_idx 42 --pruning_ratio 0.02042
