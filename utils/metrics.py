import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch


    
def subsample(ima, seed=None):
    ima = np.array(ima.cpu())
    stack = np.array([ima[:, :-1:2,:-1:2], ima[:, :-1:2,1::2], ima[:, 1::2,:-1:2], ima[:, 1::2,1::2]])
    stack = stack.transpose((1, 2, 3, 0))
    
    s = list(stack.shape)
    stack = stack.reshape((s[0]*s[1]*s[2], s[3]))
    if seed is not None:
        np.random.seed(seed)
    [np.random.shuffle(x) for x in stack]
    stack = torch.Tensor(stack.reshape(s))
    return stack[:,:,:,0], stack[:,:,:,1], stack[:,:,:,2], stack[:,:,:,3]

def uMSE_uPSNR(ds, net, seed=0):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    uMSE_k = 0
    uPSNR_k = 0
    for k in range(len(ds)):
        factor = torch.max(ds[k]).cpu()
        y, a, b, c = subsample(ds[k], seed)    
        with torch.no_grad():
            fy = net(y.to(device).unsqueeze(0))
            
            fy = fy[0][0].cpu()
            
        
        if a.shape[0] > 1:
            a = a[int(len(a)/2)]
            b = b[int(len(b)/2)]
            c = c[int(len(c)/2)]
        
        fy, a, b, c = fy/factor, a/factor, b/factor, c/factor
        
        mse = torch.mean((fy-a)**2).item()
        N = torch.mean((b-c)**2).item()/2
        
        uMSE_k = uMSE_k + mse-N
        uPSNR_k = uPSNR_k + -10. * np.log(mse-N) / np.log(10.)

    return uMSE_k/k, uPSNR_k/k


def ssim(clean, noisy, normalized=True, raw=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.] else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    
    factor = np.maximum(np.max(clean), np.max(noisy))
    
    clean = clean/factor
    noisy = noisy/factor    

    if raw:
        noisy = (np.uint16(noisy*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240)
    
    return np.array([structural_similarity(c, n, multichannel=True) for c, n in zip(clean, noisy)]).mean()


def psnr(clean, noisy, normalized=True, raw=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)

    factor = np.maximum(np.max(clean), np.max(noisy))
    
    clean = clean/factor
    noisy = noisy/factor    
        
    return np.array([peak_signal_noise_ratio(c, n, data_range = 1.0) for c, n in zip(clean, noisy)]).mean()


def mse(clean, noisy, normalized=True, raw=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)

    factor = np.maximum(np.max(clean), np.max(noisy))
    
    clean = clean/factor
    noisy = noisy/factor    
            
    return np.mean((clean-noisy)**2)
