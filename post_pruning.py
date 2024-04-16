# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import glob
import time

from efficiency.mac import compute_mask_mac
from prune.new_fisher import collect_mask_grads
from prune.rearrange import rearrange_mask
from prune.search import search_mac
from prune.new_rescale import rescale_mask
from prune_utils.schedule import get_pruning_schedule


# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/npy/CT_Abd",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth"
)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=0)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("--device", type=str, default="cuda")

parser.add_argument("--constraint", type=float, required=True,
    help="MAC/latency constraint relative to the original model",
)
args = parser.parse_args()

# %% set up dataset
tr_dataset = NpyDataset("/w/331/vyu/data")
sample_dataloader = DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model

class FakeConfig():
    def __init__(self, num_hidden_layers, num_attention_heads, intermediate_size, hidden_size):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
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
        return ori_res_masks


def main():
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    mask_decoder = sam_model.mask_decoder.to(device)
    prompt_encoder = sam_model.prompt_encoder.to(device)
    # mask_decoder = sam_model.mask_decoder
    # prompt_encoder = sam_model.prompt_encoder

    # Prepare the model
    # model = sam_model.image_encoder
    model = sam_model.image_encoder.cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # create fake config
    num_hidden_layers = len(model.blocks)
    num_attention_heads = model.blocks[0].attn.num_heads
    intermediate_size = model.blocks[0].mlp.lin1.out_features
    hidden_size = model.blocks[0].attn.proj.out_features
    config = FakeConfig(num_hidden_layers, num_attention_heads, intermediate_size, hidden_size)

    # full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads)
    # full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size)
    full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).cuda()
    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).cuda()

    # print(config.num_hidden_layers, config.num_attention_heads, config.intermediate_size, config.hidden_size)

    start = time.time()
    # Search the optimal mask
    head_grads, neuron_grads = collect_mask_grads(
        model,
        full_head_mask,
        full_neuron_mask,
        sample_dataloader,
        mask_decoder,
        prompt_encoder,
        device
    )
    teacher_constraint = get_pruning_schedule(target=args.constraint, num_iter=2)[0]
    # print(head_grads, neuron_grads)
    # seq_len = num patches
    seq_len = model.img_size // (model.patch_embed.proj.stride[0])
    print("-----searching mask-----")
    teacher_head_mask, teacher_neuron_mask = search_mac(
        config,
        head_grads,
        neuron_grads,
        seq_len,
        teacher_constraint,
    )
    head_mask, neuron_mask = search_mac(
        config,
        head_grads,
        neuron_grads,
        seq_len,
        args.constraint,
    )
    pruned_mac, orig_mac = compute_mask_mac(head_mask, neuron_mask, seq_len, config.hidden_size)
    print(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")

    # Rearrange the mask
    print("-----rearranging mask-----")
    head_mask = rearrange_mask(head_mask, head_grads)
    neuron_mask = rearrange_mask(neuron_mask, neuron_grads)
    
    # Rescale the mask by solving a least squares problem
    print("-----rescaling mask-----")
    # TODO: modify
    head_mask, neuron_mask = rescale_mask(
        model,
        config,
        teacher_head_mask,
        teacher_neuron_mask,
        head_mask,
        neuron_mask,
        sample_dataloader
    )

    # Print the pruning time
    end = time.time()
    print(f"{args.task_name} Pruning time (s): {end - start}")

    # Save the masks
    torch.save(head_mask, os.path.join(args.output_dir, "head_mask.pt"))
    torch.save(neuron_mask, os.path.join(args.output_dir, "neuron_mask.pt"))


if __name__ == "__main__":
    main()
