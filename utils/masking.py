import torch


def random_patch_mask(images, mask_ratio=0.5):
    """
    Randomly masks image patches by zeroing regions.
    """
    B, C, H, W = images.shape
    mask = torch.rand(B, 1, H, W, device=images.device)
    mask = (mask > mask_ratio).float()
    return images * mask, mask
