# /utils/metrics.py
import torch

def psnr(pred, target, max_pixel=1.0, eps=1e-10, reduction='mean'):
    """
    Calculates the Peak Signal-to-Noise Ratio between two images.

    Args:
        pred (Tensor): Predicted image tensor.
        target (Tensor): Target image tensor.
        max_pixel (float): Maximum possible pixel value of the image.
        eps (float): Small value to ensure numerical stability.
        reduction (str): Specifies the reduction to apply to the output:
            'mean' | 'none'. 'mean' calculates the mean PSNR over the batch,
            and 'none' applies no reduction, returning individual PSNR values.

    Returns:
        float or ndarray: Mean PSNR of the batch if reduction is 'mean',
        or an array of PSNR values for each image in the batch if reduction is 'none'.
    """

    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr_val = 20 * torch.log10(max_pixel / torch.sqrt(mse + eps))

    if reduction == 'mean':
        return psnr_val.mean().cpu().item()
    elif reduction == 'none':
        return psnr_val.cpu().numpy()
    else:
        raise ValueError("Invalid reduction mode. Supported modes are 'mean' and 'none'.")