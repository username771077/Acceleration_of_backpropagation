import torch
import torch.nn as nn
from transformers import PreTrainedModel

from sparse_method_triton import SparseMethodLinearIntermediate, SparseMethodLinearOutput

def replace_bert_layers(model: PreTrainedModel, layers_to_modify: list = None):
    """Replaces nn.Linear in BERT FFN layers with SparseMethodLinear layers."""
    action = "Replacing ALL" if layers_to_modify is None else f"Replacing layers {layers_to_modify} in"
    print(f"! {action} BERT FFN layers with SparseMethodLinear layers !")
    device = next(model.parameters()).device; dtype = next(model.parameters()).dtype
    encoder = None
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"): encoder = model.bert.encoder
    elif hasattr(model, "encoder"): encoder = model.encoder
    else: print("Warning: Could not find standard BERT encoder structure."); return model

    layers_replaced_count = 0
    if not hasattr(encoder, "layer"): print("Warning: Encoder does not have 'layer' attribute."); return model

    modify_set = set(layers_to_modify) if layers_to_modify is not None else None

    for i, layer in enumerate(encoder.layer):
        should_modify = (modify_set is None or i in modify_set)
        intermediate_module = getattr(layer, "intermediate", None)
        if intermediate_module:
            dense_layer = getattr(intermediate_module, "dense", None)
            if dense_layer and isinstance(dense_layer, nn.Linear):
                if should_modify:
                    orig = dense_layer; h, t = orig.weight.shape 
                    new = SparseMethodLinearIntermediate(t, h, i, (orig.bias is not None), device=device, dtype=dtype).from_linear(orig)
                    layer.intermediate.dense = new; layers_replaced_count +=1

        output_module = getattr(layer, "output", None)
        if output_module:
            dense_layer = getattr(output_module, "dense", None)
            if dense_layer and isinstance(dense_layer, nn.Linear):
                 if should_modify:
                    orig = dense_layer; t_out, h_out = orig.weight.shape 
                    new = SparseMethodLinearOutput(h_out, t_out, i, (orig.bias is not None), device=device, dtype=dtype).from_linear(orig)
                    layer.output.dense = new; layers_replaced_count += 1

    if layers_replaced_count > 0: print(f"! Layer replacement complete ({layers_replaced_count} Linear layers modified) !")
    elif layers_to_modify is not None: print(f"! No layers were replaced for specified indices {layers_to_modify}. Check indices/structure. !")
    else: print("! No layers were replaced. Check model structure or replacement logic. !")
    return model

def replace_roberta_layers(model: PreTrainedModel, layers_to_modify: list = None):
    """Replaces nn.Linear in RoBERTa FFN layers with SparseMethodLinear layers."""
    action = "Replacing ALL" if layers_to_modify is None else f"Replacing layers {layers_to_modify} in"
    print(f"! {action} RoBERTa FFN layers with SparseMethodLinear layers !")
    device = next(model.parameters()).device; dtype = next(model.parameters()).dtype
    encoder = None
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"): encoder = model.roberta.encoder
    elif hasattr(model, "encoder"): encoder = model.encoder
    else: print("Warning: Could not find standard RoBERTa encoder structure."); return model

    layers_replaced_count = 0
    if not hasattr(encoder, "layer"): print("Warning: Encoder does not have 'layer' attribute."); return model

    modify_set = set(layers_to_modify) if layers_to_modify is not None else None

    for i, layer in enumerate(encoder.layer):
        should_modify = (modify_set is None or i in modify_set)
        intermediate_module = getattr(layer, "intermediate", None)
        if intermediate_module:
            dense_layer = getattr(intermediate_module, "dense", None)
            if dense_layer and isinstance(dense_layer, nn.Linear):
                if should_modify:
                    orig = dense_layer; h, t = orig.weight.shape 
                    new = SparseMethodLinearIntermediate(t, h, i, (orig.bias is not None), device=device, dtype=dtype).from_linear(orig)
                    layer.intermediate.dense = new; layers_replaced_count += 1

        output_module = getattr(layer, "output", None)
        if output_module:
            dense_layer = getattr(output_module, "dense", None)
            if dense_layer and isinstance(dense_layer, nn.Linear):
                if should_modify:
                    orig = dense_layer; t_out, h_out = orig.weight.shape                      
                    new = SparseMethodLinearOutput(h_out, t_out, i, (orig.bias is not None), device=device, dtype=dtype).from_linear(orig)
                    layer.output.dense = new; layers_replaced_count += 1

    if layers_replaced_count > 0: print(f"! Layer replacement complete ({layers_replaced_count} Linear layers modified) !")
    elif layers_to_modify is not None: print(f"! No layers were replaced for specified indices {layers_to_modify}. Check indices/structure. !")
    else: print("! No layers were replaced. Check model structure or replacement logic. !")
    return model

print("Model utility functions defined.")
print("-" * 40)