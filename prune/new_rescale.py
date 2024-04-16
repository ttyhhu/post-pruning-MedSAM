from tqdm import tqdm
import torch

from prune_utils.run_model_with_mask import run_blk_with_head_mask
from prune_utils.linalg import lsmr_cupy_solver
from prune_utils.new_arch import (
    get_layers,
    get_mha_proj,
    get_ffn2,
    hijack_input,
    MaskNeurons,
    collect_layer_inputs,
)


@torch.no_grad()
def get_mha_lstsq(
    model,
    config,
    teacher_inputs,
    teacher_neuron_mask,
    student_inputs,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
):
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    nonzero_heads = student_head_mask[layer_idx].nonzero().flatten()
    num_nonzero_heads = nonzero_heads.shape[0]

    layer = get_layers(model)[layer_idx]
    mha_proj = get_mha_proj(model, layer_idx)
    weights_per_head = mha_proj.weight.t().view(num_attention_heads, -1, hidden_size)
    weights_per_head = weights_per_head.index_select(dim=0, index=nonzero_heads)

    inputs = []
    handle = hijack_input(mha_proj, inputs)

    # ATA = torch.zeros(num_nonzero_heads + 1, num_nonzero_heads + 1)
    # ATB = torch.zeros(num_nonzero_heads + 1)
    ATA = torch.zeros(num_nonzero_heads + 1, num_nonzero_heads + 1).cuda()
    ATB = torch.zeros(num_nonzero_heads + 1).cuda()

    model.eval()
    for teacher_batch, student_batch in zip(teacher_inputs, student_inputs):
        student_batch[1] = student_head_mask[layer_idx]

        # Get the outputs of the teacher model
        with MaskNeurons(model, teacher_neuron_mask):
            run_blk_with_head_mask(layer, teacher_batch[1], teacher_batch[0])
        hidden_states, input_tensor = inputs.pop(0)
        teacher_output = mha_proj(hidden_states) + input_tensor

        # Get the outputs of the student model
        with MaskNeurons(model, student_neuron_mask):
            run_blk_with_head_mask(layer, student_batch[1], student_batch[0])
        hidden_states, input_tensor = inputs.pop(0)

        hidden_states = hidden_states.view(-1, num_attention_heads, attention_head_size)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = hidden_states.index_select(dim=0, index=nonzero_heads)

        outputs_per_head = hidden_states @ weights_per_head
        outputs_per_head = outputs_per_head.view(num_nonzero_heads, -1)

        A = outputs_per_head.t()
        A = torch.cat([A, torch.ones(A.shape[0], 1).cuda()], dim=1)
        # A = torch.cat([A, torch.ones(A.shape[0], 1)], dim=1)
        B = teacher_output - mha_proj.bias - input_tensor
        B = B.flatten()

        ATA += A.t() @ A
        ATB += A.t() @ B

    handle.remove()
    return ATA, ATB


@torch.no_grad()
def get_ffn_lstsq(
    model,
    config,
    teacher_inputs,
    teacher_neuron_mask,
    student_inputs,
    student_head_mask,
    student_neuron_mask,
    layer_idx,
):
    layer = get_layers(model)[layer_idx]
    ffn2 = get_ffn2(model, layer_idx)
    weights_per_neuron = ffn2.weight.t()

    nonzero_neurons = student_neuron_mask[layer_idx].nonzero().flatten()
    num_neurons = nonzero_neurons.shape[0]
    weights_per_neuron = weights_per_neuron.index_select(dim=0, index=nonzero_neurons)
    W = weights_per_neuron @ weights_per_neuron.t()

    inputs = []
    handle = hijack_input(ffn2, inputs)

    # ATA = torch.zeros(num_neurons, num_neurons)
    # ATB = torch.zeros(num_neurons)
    ATA = torch.zeros(num_neurons, num_neurons).cuda()
    ATB = torch.zeros(num_neurons).cuda()

    model.eval()
    for teacher_batch, student_batch in zip(teacher_inputs, student_inputs):
        student_batch[1] = student_head_mask[layer_idx]

        # Get the outputs of the teacher model
        with MaskNeurons(model, teacher_neuron_mask):
            run_blk_with_head_mask(layer, teacher_batch[1], teacher_batch[0])
        hidden_states, input_tensor = inputs.pop(0)
        teacher_output = ffn2(hidden_states) + input_tensor
        teacher_output = teacher_output[:, 0, :]

        # Get the outputs of the student model
        with MaskNeurons(model, student_neuron_mask):
            run_blk_with_head_mask(layer, student_batch[1], student_batch[0])
        hidden_states, input_tensor = inputs.pop(0)
        hidden_states = hidden_states[:, 0, :]
        input_tensor = input_tensor[:, 0, :]

        hidden_states = hidden_states.t()
        hidden_states = hidden_states.index_select(dim=0, index=nonzero_neurons)

        ATA += W * (hidden_states @ hidden_states.t())

        B = teacher_output - ffn2.bias - input_tensor
        ATB += (hidden_states.unsqueeze(1) @ (weights_per_neuron @ B.t()).unsqueeze(2)).squeeze()

    handle.remove()
    return ATA, ATB


@torch.no_grad()
def rescale_mask(
    model,
    config,
    teacher_head_mask,
    teacher_neuron_mask,
    student_head_mask,
    student_neuron_mask,
    dataloader
):
    num_hidden_layers = config.num_hidden_layers
    rescaled_head_mask = student_head_mask.clone()
    rescaled_neuron_mask = student_neuron_mask.clone()

    for layer_idx in tqdm(range(num_hidden_layers)):
        teacher_inputs = collect_layer_inputs(
            model,
            teacher_head_mask,
            teacher_neuron_mask,
            layer_idx,
            prev_inputs=dataloader if layer_idx == 0 else teacher_inputs,
        )
        student_inputs = collect_layer_inputs(
            model,
            rescaled_head_mask,
            rescaled_neuron_mask,
            layer_idx,
            prev_inputs=dataloader if layer_idx == 0 else student_inputs,
        )

        if torch.count_nonzero(student_head_mask[layer_idx]) != 0 and layer_idx != 0:
            # print("head mask in")
            ATA, ATB = get_mha_lstsq(
                model,
                config,
                teacher_inputs,
                teacher_neuron_mask,
                student_inputs,
                rescaled_head_mask,
                rescaled_neuron_mask,
                layer_idx,
            )
            scale_factor, success = lsmr_cupy_solver(ATA, ATB)
            if not success:
                break
            scale_factor = scale_factor[:-1]
            if scale_factor.max() > 10 or scale_factor.min() < -10:
                break
            nonzero_heads = rescaled_head_mask[layer_idx].nonzero().flatten()
            for index, scale in zip(nonzero_heads, scale_factor):
                rescaled_head_mask[layer_idx][index] *= scale

        if torch.count_nonzero(student_neuron_mask[layer_idx]) != 0:
            # print("ffn mask in")
            ATA, ATB = get_ffn_lstsq(
                model,
                config,
                teacher_inputs,
                teacher_neuron_mask,
                student_inputs,
                rescaled_head_mask,
                rescaled_neuron_mask,
                layer_idx,
            )
            scale_factor, success = lsmr_cupy_solver(ATA, ATB)
            if not success:
                break
            if scale_factor.max() > 10 or scale_factor.min() < -10:
                break
            nonzero_neurons = rescaled_neuron_mask[layer_idx].nonzero().flatten()
            for index, scale in zip(nonzero_neurons, scale_factor):
                rescaled_neuron_mask[layer_idx][index] *= scale

    return rescaled_head_mask, rescaled_neuron_mask
