import torch


def random_patch_mask(images, mask_ratio=0.5, patch_size=16):
    """
    images: (B, 3, H, W)
    returns:
        masked_images: same shape
        mask: binary mask (1 = masked)
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_h * num_patches_w
    num_mask = int(mask_ratio * total_patches)

    mask = torch.zeros((B, total_patches), device=images.device)

    for i in range(B):
        idx = torch.randperm(total_patches)[:num_mask]
        mask[i, idx] = 1

    mask = mask.view(B, num_patches_h, num_patches_w)

    masked_images = images.clone()

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            masked_images[
                :, :,
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size
            ] *= (1 - mask[:, i, j].view(-1, 1, 1, 1))

    return masked_images, mask

