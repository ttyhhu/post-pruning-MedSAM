import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import monai

from prune_utils.new_arch import apply_neuron_mask
from prune_utils.run_model_with_mask import run_model_with_head_mask

def collect_mask_grads(model, head_mask, neuron_mask, dataloader, mask_decoder, prompt_encoder, device):
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    head_mask.requires_grad_(True)
    neuron_mask.requires_grad_(True)

    handles = apply_neuron_mask(model, neuron_mask)

    model.eval()
    head_grads = []
    neuron_grads = []
    for step, (image, gt2D, boxes, _) in enumerate(tqdm(dataloader)):
        image, gt2D = image.cuda(), gt2D.cuda()
        outputs = run_model_with_head_mask(model, head_mask, image)
        
        boxes_np = boxes.detach().cpu().numpy()
        with torch.no_grad():
            box_torch = torch.as_tensor(boxes_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            sparse_embeddings, dense_embeddings = prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = mask_decoder(
            image_embeddings=outputs,  # (B, 256, 64, 64)
            image_pe=prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        loss = seg_loss(ori_res_masks, gt2D) + ce_loss(ori_res_masks, gt2D.float())
        loss.backward()

        head_grads.append(head_mask.grad.detach())
        head_mask.grad = None

        neuron_grads.append(neuron_mask.grad.detach())
        neuron_mask.grad = None

    for handle in handles:
        handle.remove()
    head_mask.requires_grad_(False)
    neuron_mask.requires_grad_(False)

    head_grads = torch.stack(head_grads, dim=0)
    neuron_grads = torch.stack(neuron_grads, dim=0)
    return head_grads, neuron_grads


@torch.no_grad()
def compute_fisher_info(grads):
    fisher_info = grads.pow(2).sum(dim=0)
    return fisher_info
