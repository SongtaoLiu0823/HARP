import torch 
import torch.nn as nn 
from .layerwrapper import WrappedGPT, BiasGPT
from .data import get_loaders 
from tqdm import tqdm

# create a dictionary to map the method name to the function
"""
    'IFV': Input Feature Variance
    'WIFV': Weighted Input Feature Variance
    'WIFN': Weighted Input Feature Norm
"""
metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
}


def calculate_model_params(model):
    """
    Calculate the total number of parameters in layers of the model.
    
    Args:
        model (nn.Module): The model to analyze.
        
    Returns:
        tuple: Total number of parameters parameters in the model.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)  # Find all layers in the current layer module

        for name in subset:
            W = subset[name].weight.data
            total_params += W.numel()  # Add the number of elements in the weight tensor
            if subset[name].bias is not None:
                total_params += subset[name].bias.data.numel()  # Add bias parameters if present
    
    model.config.use_cache = use_cache
    return total_params


def calculate_query_params(model):
    """
    Calculate the number of parameters in q_proj layers of the model.

    Args:
        model (nn.Module): The model to analyze.

    Returns:
        int: Total number of parameters in q_proj layers.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    query_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)  # Find all layers in the current layer module

        for name in subset:
            W = subset[name].weight.data
            if 'q_proj' in name:  # Check if the layer belongs to k_proj
                query_params += W.numel()
                if subset[name].bias is not None:
                    query_params += subset[name].bias.data.numel()  # Add bias parameters if present

    model.config.use_cache = use_cache
    return query_params


def calculate_key_params(model):
    """
    Calculate the number of parameters in k_proj layers of the model.

    Args:
        model (nn.Module): The model to analyze.

    Returns:
        int: Total number of parameters in k_proj layers.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    key_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)  # Find all layers in the current layer module

        for name in subset:
            W = subset[name].weight.data
            if 'k_proj' in name:  # Check if the layer belongs to k_proj
                key_params += W.numel()
                if subset[name].bias is not None:
                    key_params += subset[name].bias.data.numel()  # Add bias parameters if present

    model.config.use_cache = use_cache
    return key_params


def calculate_value_params(model):
    """
    Calculate the number of parameters in v_proj layers of the model.

    Args:
        model (nn.Module): The model to analyze.

    Returns:
        int: Total number of parameters in v_proj layers.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    value_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)  # Find all layers in the current layer module

        for name in subset:
            W = subset[name].weight.data
            if 'v_proj' in name:  # Check if the layer belongs to v_proj
                value_params += W.numel()
                if subset[name].bias is not None:
                    value_params += subset[name].bias.data.numel()  # Add bias parameters if present

    model.config.use_cache = use_cache
    return value_params


def calculate_output_params(model):
    """
    Calculate the number of parameters in o_proj layers of the model.

    Args:
        model (nn.Module): The model to analyze.

    Returns:
        int: Total number of parameters in o_proj layers.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    output_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)  # Find all layers in the current layer module

        for name in subset:
            W = subset[name].weight.data
            if 'o_proj' in name:  # Check if the layer belongs to v_proj
                output_params += W.numel()
                if subset[name].bias is not None:
                    output_params += subset[name].bias.data.numel()  # Add bias parameters if present

    model.config.use_cache = use_cache
    return output_params


def analyze_linear_layers(model):
    """
    Analyze the shapes of all linear layers in the model and print their details.

    Args:
        model (nn.Module): The model to analyze.

    Returns:
        None
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)  # Find all layers in the current layer module

        print(f"Layer {i}:")
        for name in subset:
            W = subset[name].weight.data
            print(f"  {name}: weight shape {W.shape}")
            if subset[name].bias is not None:
                print(f"  {name}: bias shape {subset[name].bias.data.shape}")

    model.config.use_cache = use_cache


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def prepare_calibration_input(model, dataloader, device):
    """
    Prepare inputs for model calibration. 
    
    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded. 
        
    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((2048, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def compress(layer, mlp_mask, mlp_mean_inp, device, bias=False, args=None):
    """
    Compress a model layer by masking or pruning based on the given masks.
    
    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        
    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    # Real Pruning

    # MLP Weight Pruning
    # Prune the up and gate projection weights
    layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
    layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]

    if hasattr(layer.mlp.up_proj, 'bias') and layer.mlp.up_proj.bias is not None:
        layer.mlp.up_proj.bias.data = layer.mlp.up_proj.bias.data[torch.where(mlp_mask)[0]]
    if hasattr(layer.mlp.gate_proj, 'bias') and layer.mlp.gate_proj.bias is not None:
        layer.mlp.gate_proj.bias.data = layer.mlp.gate_proj.bias.data[torch.where(mlp_mask)[0]]
    
    # Update output dimensions of up and gate projections based on the mlp mask
    layer.mlp.up_proj.out_features = mlp_mask.sum().item()
    layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
    
    output_weight = layer.mlp.down_proj.weight.data
    layer.mlp.intermediate_size = mlp_mask.sum().item()
    if bias:
        # Add the additional bias to compensate for the loss
        output_bias = ((mlp_mean_inp.to(device) * ~mlp_mask.to(device)) @ output_weight.T)
        
    # Prune the down projection weight
    output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]  
    
    if bias:
        # Re-initialize the Linear layer with new shape and bias
        layer.mlp.down_proj.in_features = mlp_mask.sum().item()
        # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
        layer.mlp.down_proj.bias.data = output_bias
        
    # Assign the pruned weights
    layer.mlp.down_proj.weight.data = output_weight
    
    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()


def prune_flap(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Our FLAP Pruning.
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    layers = model.model.layers

    mlp_metric_list = []
    mlp_baseline_inp_list = []
    mlp_mask = []
        
    # Split into sub-problems, separate statistics for each module
    for i in tqdm(range(args.start_pruning_layer_idx, args.end_pruning_layer_idx), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BiasGPT(subset[name], args.metrics)            

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            W_metric = metrics[args.metrics](wrapped_layers, subset, name)
            mlp_metric_list.append(W_metric.cpu())
            mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.bfloat16))
            wrapped_layers[name].free()

        inps, outs = outs, inps # Use the original output as input to the next layer
        torch.cuda.empty_cache()

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

    mlp_metric = torch.stack(mlp_metric_list)
    mlp_metric = standarlization(mlp_metric)
    
    prune_metric = torch.cat([mlp_metric.view(-1)])
    sorted_prune, indices = torch.sort(prune_metric, descending=True)
    compression_weight = torch.ones_like(indices)
    threshold = sorted_prune[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*(1 - args.pruning_ratio)))]
    mlp_mask = (mlp_metric > threshold)

    
    for idx in range(0, args.end_pruning_layer_idx-args.start_pruning_layer_idx):
        compress(model.model.layers[args.start_pruning_layer_idx+idx], mlp_mask[idx], mlp_baseline_inp_list[idx], device, bias=args.bias, args=args)
                
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_wanda_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Wanda on structured pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=128,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(args.start_pruning_layer_idx, args.end_pruning_layer_idx):
        layer = layers[i]
        subset = {}
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            W_metric = W_metric.mean(axis=0)
            thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.pruning_ratio)].cpu()
            W_mask = (W_metric>=thresh)
            compress(layer, W_mask, None, device, bias=args.bias, args=args)
          
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps # the pruned output as input to the next layer
        
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


def prune_magnitude_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Magnitude Pruning on structured pruning.
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    layers = model.model.layers 

    for i in range(args.start_pruning_layer_idx, args.end_pruning_layer_idx):
        layer = layers[i]
        subset = {}
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.norm(subset[name].weight.data, dim=0).double()

            thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.pruning_ratio)].cpu()
            W_mask = (W_metric>=thresh)
            compress(layer, W_mask, None, device, bias=args.bias, args=args)

