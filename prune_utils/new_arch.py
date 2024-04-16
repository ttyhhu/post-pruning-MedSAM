# reference: retraining-free-pruning
import torch
from tqdm import tqdm
from prune_utils.run_model_with_mask import run_model_with_head_mask, run_blk_with_head_mask


class MaskNeurons:
    def __init__(self, model, neuron_mask):
        self.handles = apply_neuron_mask(model, neuron_mask)

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        for handle in self.handles:
            handle.remove()


def get_layers(model):
    layers = model.blocks
    return layers

def get_ffn2(model, index):
    layer = get_layers(model)[index]
    ffn2 = layer.mlp.lin2
    return ffn2

def get_mha_proj(model, index):
    layer = get_layers(model)[index]
    mha_proj = layer.attn.proj
    return mha_proj

def hijack_input(module, list_to_append):
    hook = lambda _, inputs: list_to_append.append(inputs)
    handle = module.register_forward_pre_hook(hook)
    return handle

def register_mask(module, mask):
    hook = lambda _, inputs: (inputs[0] * mask)
    handle = module.register_forward_pre_hook(hook)
    return handle


def apply_neuron_mask(model, neuron_mask):
    num_hidden_layers = neuron_mask.shape[0]
    handles = []
    for layer_idx in range(num_hidden_layers):
        ffn2 = get_ffn2(model, layer_idx)
        handle = register_mask(ffn2, neuron_mask[layer_idx])
        handles.append(handle)
    return handles


@torch.no_grad()
def collect_layer_inputs(
    model,
    head_mask,
    neuron_mask,
    layer_idx,
    prev_inputs,
):
    layers = get_layers(model)
    target_layer = layers[layer_idx]

    inputs = []
    if layer_idx == 0:
        encoder = model
        layers = get_layers(model)
        encoder.layers = layers[:1]

        handle = hijack_input(target_layer, inputs)
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(prev_inputs)):
            image, gt2D = image.cuda(), gt2D.cuda()
            with MaskNeurons(model, neuron_mask):
                run_model_with_head_mask(model, head_mask, image)

        handle.remove()
        encoder.layers = layers
        inputs = [list(x) for x in inputs]
    else:
        prev_layer = layers[layer_idx - 1]
        
        for step, (x, mask) in enumerate(tqdm(prev_inputs)):
            image, gt2D = image.cuda(), gt2D.cuda()
            layer_head_mask = head_mask[layer_idx - 1]
            with MaskNeurons(model, neuron_mask):
                prev_output = run_blk_with_head_mask(prev_layer, layer_head_mask, x)
            inputs.append([prev_output, head_mask[layer_idx]])

    return inputs
