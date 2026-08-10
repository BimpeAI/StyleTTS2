from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

try:
    from monotonic_align import mask_from_lens
except ImportError:
    # PyPI `monotonic_align` lacks this; Resemble AI fork has it.
    # Provide a local fallback so training/inference still run.
    def mask_from_len(lens: torch.Tensor, max_len=None):
        if max_len is None:
            max_len = lens.max()
        index = torch.arange(max_len, device=lens.device).view(1, -1)
        return index < lens.unsqueeze(1)

    def mask_from_lens(similarity: torch.Tensor, symbol_lens: torch.Tensor, mel_lens: torch.Tensor):
        _, S, T = similarity.size()
        mask_S = mask_from_len(symbol_lens, S)
        mask_T = mask_from_len(mel_lens, T)
        mask_ST = mask_S.unsqueeze(2) * mask_T.unsqueeze(1)
        return mask_ST.to(similarity)


def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.exp(torch.clamp(x * std + mean, min=-80, max=80))
    x = torch.log(x.norm(dim=dim) + 1e-6)
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message, flush=True)


def is_improved(current, best, min_delta=0.0):
    """True if current val loss is better than best by at least min_delta."""
    if current is None or (isinstance(current, float) and current != current):  # NaN
        return False
    return current < (best - min_delta)


def prune_other_checkpoints(log_dir, glob_pattern, keep_paths):
    """Delete files matching glob_pattern except those in keep_paths."""
    import glob
    import os
    import os.path as osp

    keep = {osp.abspath(p) for p in keep_paths if p}
    for path in glob.glob(osp.join(log_dir, glob_pattern)):
        if osp.abspath(path) not in keep:
            try:
                os.remove(path)
            except OSError:
                pass


def save_best_checkpoint(log_dir, state, epoch_prefix, epoch, best_name):
    """
    Save current best as epoch_{prefix}_XXXXX.pth and a stable best_name copy.
    Returns paths (epoch_path, best_path).
    """
    import os
    import os.path as osp
    import torch

    os.makedirs(log_dir, exist_ok=True)
    epoch_path = osp.join(log_dir, f'epoch_{epoch_prefix}_%05d.pth' % epoch)
    best_path = osp.join(log_dir, best_name)
    torch.save(state, epoch_path)
    torch.save(state, best_path)
    prune_other_checkpoints(log_dir, f'epoch_{epoch_prefix}_*.pth', [epoch_path, best_path])
    return epoch_path, best_path


def append_epoch_metrics(csv_path, row):
    """Append one epoch metrics row; write header if the file is new."""
    import csv
    import os
    import os.path as osp

    os.makedirs(osp.dirname(osp.abspath(csv_path)) or '.', exist_ok=True)
    write_header = not osp.isfile(csv_path) or osp.getsize(csv_path) == 0
    fieldnames = list(row.keys())
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
