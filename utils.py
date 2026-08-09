import torch.nn as nn 


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

