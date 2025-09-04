# ComfyUI Multi-Analysis Heatmaps (Extended & Optimized, Robust I/O)
# Version: 3.0.1
# Parallel outputs:
#  - SSIM, DSSIM, MS-SSIM
#  - LPIPS (approx via VGG16 features, optional)
#  - Per-pixel abs diff, Residual (signed)
#  - FFT spectra (orig, upscaled, diff)
#  - ΔE2000 heatmap + L*, a*, b* diffs (Lab)
#  - Gradient diff (Scharr) + Edge IoU ("Canny-like")
#  - Wavelet band diff (Haar, multi-level)
#  - Denoiser residuum (Gaussian high-pass)
#  - Round-trip stability map (downscale->upscale)
#
# Perf:
#  - dtype-safe fp32, autocast off, no_grad
#  - cached Gaussian kernels & VGG16 features
#
# No external deps beyond torch/torchvision (optional for LPIPS).
# Author: chatgpt

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================= Node Meta =======================
NODE_NAME = "Image Analysis (SSIM/LPIPS/FFT/ΔE2000)"
NODE_CATEGORY = "Image/Analysis"
NODE_AUTHOR = "chatgpt"
NODE_VERSION = "3.0.1"

# ======================= Caches ==========================
_GAUSS2D_CACHE = {}   # (dev, idx, dtype, ch, win, sigma) -> [C,1,win,win]
_GAUSS1D_CACHE = {}   # (dev, idx, dtype, win, sigma) -> [1,1,1,W]
_VGG_CACHE = {}       # (dev, idx) -> {"model": VGG16Features, "mean": t, "std": t}

def _dev_key(device: torch.device, dtype=None):
    return (device.type, device.index if device.index is not None else -1, str(dtype) if dtype else "")

# ======================= Tensor helpers ==================
def _squeeze_to_4d(img: torch.Tensor) -> torch.Tensor:
    """
    Squeeze überzählige Singleton-Dimensionen (size==1), bis der Tensor <= 4D ist.
    Bewahrt Batch/Channels, weil wir nur >4D-Dims reduzieren.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"_squeeze_to_4d expected torch.Tensor, got {type(img)}")

    t = img
    # so lange rank > 4 und es gibt mind. eine 1er-Dimension -> squeeze genau EINE pro Iteration
    while t.dim() > 4 and 1 in t.shape:
        # squeeze die erste Singleton-Dimension > 0 (Batch lassen wir typischerweise in Ruhe,
        # aber bei rank>4 ist es egal – Batch ist i.d.R. dim 0 und bleibt 1; wir können auch die erste 1 nehmen)
        for d, s in enumerate(t.shape):
            if s == 1:
                t = t.squeeze(d)
                break
    return t


def to_bchw(img: torch.Tensor) -> torch.Tensor:
    """
    Robust conversion to [B,C,H,W].
    Akzeptiert u.a. [B,H,W,C], [B,C,H,W], [H,W,C], [C,H,W], [H,W] und
    auch höhere Ränge mit Singleton-Dims wie [B,C,1,H,W] – diese werden
    vorab auf 4D gesqueezed.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"to_bchw expected torch.Tensor, got {type(img)}")

    x = _squeeze_to_4d(img)

    if x.dim() == 4:
        # [B,H,W,C] -> [B,C,H,W]
        if x.shape[-1] in (1, 3):
            return x.permute(0, 3, 1, 2).contiguous()
        # assume already [B,C,H,W]
        return x.contiguous()

    if x.dim() == 3:
        # [H,W,C] -> [1,C,H,W]
        if x.shape[-1] in (1, 3):
            return x.permute(2, 0, 1).unsqueeze(0).contiguous()
        # [C,H,W] -> [1,C,H,W]
        if x.shape[0] in (1, 3):
            return x.unsqueeze(0).contiguous()
        # unbekannt, behandle als [C,H,W]
        return x.unsqueeze(0).contiguous()

    if x.dim() == 2:
        # [H,W] -> [1,1,H,W]
        return x.unsqueeze(0).unsqueeze(0).contiguous()

    raise ValueError(f"to_bchw expected 2D/3D/4D tensor after squeeze, got shape {tuple(x.shape)}")


def to_bhwc(img: torch.Tensor) -> torch.Tensor:
    """
    Robust conversion to [B,H,W,C].
    Akzeptiert u.a. [B,C,H,W], [B,H,W,C], [C,H,W], [H,W,C], [H,W] und
    auch höhere Ränge mit Singleton-Dims (z.B. [B,C,1,H,W]).
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"to_bhwc expected torch.Tensor, got {type(img)}")

    x = _squeeze_to_4d(img)

    if x.dim() == 4:
        # bereits BHWC?
        if x.shape[-1] in (1, 3):
            return x.contiguous()
        # sonst BCHW -> BHWC
        return x.permute(0, 2, 3, 1).contiguous()

    if x.dim() == 3:
        # [H,W,C] -> [1,H,W,C]
        if x.shape[-1] in (1, 3):
            return x.unsqueeze(0).contiguous()
        # [C,H,W] -> [1,H,W,C]
        if x.shape[0] in (1, 3):
            return x.permute(1, 2, 0).unsqueeze(0).contiguous()
        # unbekannt, behandle als [C,H,W]
        return x.permute(1, 2, 0).unsqueeze(0).contiguous()

    if x.dim() == 2:
        # [H,W] -> [1,H,W,1]
        return x.unsqueeze(0).unsqueeze(-1).contiguous()

    raise ValueError(f"to_bhwc expected 2D/3D/4D tensor after squeeze, got shape {tuple(x.shape)}")


def _ensure_bhwc3(x: torch.Tensor) -> torch.Tensor:
    """
    Für ComfyUI-Preview: sichere BHWC & 3 Kanäle.
    """
    t = to_bhwc(x)
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)
    return t


def ensure_same_device(*tensors):
    dev = None
    for t in tensors:
        if isinstance(t, torch.Tensor):
            dev = t.device
            break
    if dev is None:
        dev = torch.device("cpu")
    out = []
    for t in tensors:
        out.append(t.to(dev) if isinstance(t, torch.Tensor) else t)
    return out

def resize_like(img_bchw: torch.Tensor, target_bchw: torch.Tensor) -> torch.Tensor:
    _, _, H, W = target_bchw.shape
    return F.interpolate(img_bchw, size=(H, W), mode="bilinear", align_corners=False)

def center_crop_to_min(a: torch.Tensor, b: torch.Tensor):
    _, _, Ha, Wa = a.shape
    _, _, Hb, Wb = b.shape
    H = min(Ha, Hb); W = min(Wa, Wb)
    def crop(x):
        _, _, Hx, Wx = x.shape
        top = (Hx - H) // 2
        left = (Wx - W) // 2
        return x[:, :, top:top+H, left:left+W]
    return crop(a), crop(b)

def pad_to_max(a: torch.Tensor, b: torch.Tensor):
    _, _, Ha, Wa = a.shape
    _, _, Hb, Wb = b.shape
    H = max(Ha, Hb); W = max(Wa, Wb)
    def pad(x):
        _, _, Hx, Wx = x.shape
        pt = (H - Hx) // 2
        pb = H - Hx - pt
        pl = (W - Wx) // 2
        pr = W - Wx - pl
        return F.pad(x, (pl, pr, pt, pb), mode="constant", value=0.0)
    return pad(a), pad(b)

# ======================= Normalization & Colormaps =======
def _ramp(x, a, b):
    return torch.clamp((x - a) / (b - a), 0.0, 1.0)

def jet_colormap(v01: torch.Tensor) -> torch.Tensor:
    r = _ramp(v01, 0.35, 0.66) + _ramp(v01, 0.66, 1.0)
    g = _ramp(v01, 0.0, 0.125) + _ramp(v01, 0.125, 0.375) + _ramp(v01, 0.375, 0.66) - _ramp(v01, 0.66, 0.875)
    b = _ramp(v01, 0.0, 0.375) + _ramp(v01, 0.375, 0.66)
    return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)

def grayscale_colormap(v01: torch.Tensor) -> torch.Tensor:
    return v01.repeat(1, 3, 1, 1)

def apply_colormap(v01: torch.Tensor, mode: str = "jet") -> torch.Tensor:
    return grayscale_colormap(v01) if mode == "grayscale" else jet_colormap(v01)

def normalize01(x: torch.Tensor) -> torch.Tensor:
    xmin = x.amin(dim=(2,3), keepdim=True)
    xmax = x.amax(dim=(2,3), keepdim=True)
    span = torch.clamp(xmax - xmin, min=1e-8)
    return torch.clamp((x - xmin) / span, 0.0, 1.0)

# ======================= Gaussian Kernels =================
def gaussian2d(window_size=11, sigma=1.5, channels=3, device="cpu", dtype=torch.float32):
    key = (_dev_key(torch.device(device), dtype), channels, int(window_size), float(sigma))
    k = _GAUSS2D_CACHE.get(key)
    if k is not None and k.device == torch.device(device) and k.dtype == dtype:
        return k
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / (g.sum() + 1e-12)
    k2d = torch.outer(g, g)
    k = k2d.expand(channels, 1, window_size, window_size).contiguous()
    _GAUSS2D_CACHE[key] = k
    return k

def gaussian1d(window_size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    key = (_dev_key(torch.device(device), dtype), int(window_size), float(sigma))
    k = _GAUSS1D_CACHE.get(key)
    if k is not None and k.device == torch.device(device) and k.dtype == dtype:
        return k
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / (g.sum() + 1e-12)
    k = g.view(1, 1, 1, -1).contiguous()  # [1,1,1,W]
    _GAUSS1D_CACHE[key] = k
    return k

def gaussian_blur(img_bchw: torch.Tensor, win=11, sigma=1.5):
    # separable blur (faster), per-channel depthwise
    B, C, H, W = img_bchw.shape
    device, dtype = img_bchw.device, img_bchw.dtype
    k1 = gaussian1d(win, sigma, device, dtype)       # [1,1,1,W]
    k1 = k1.expand(C, 1, 1, -1).contiguous()         # [C,1,1,W]
    k1t = k1.transpose(2, 3).contiguous()            # [C,1,W,1]
    # horizontal
    x = F.conv2d(img_bchw, k1, groups=C, padding=(0, win//2))
    # vertical
    x = F.conv2d(x, k1t, groups=C, padding=(win//2, 0))
    return x

# ======================= SSIM / MS-SSIM ==================
def ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size=11, sigma=1.5, eps=1e-12):
    assert img1.shape == img2.shape, "SSIM input shapes must match"
    B, C, H, W = img1.shape
    device = img1.device
    dtype = img1.dtype
    C1 = torch.as_tensor((0.01 ** 2), device=device, dtype=dtype)
    C2 = torch.as_tensor((0.03 ** 2), device=device, dtype=dtype)
    win = gaussian2d(window_size, sigma, channels=C, device=device, dtype=dtype)

    mu1 = F.conv2d(img1, win, groups=C, padding=window_size//2)
    mu2 = F.conv2d(img2, win, groups=C, padding=window_size//2)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, win, groups=C, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, win, groups=C, padding=window_size//2) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, win, groups=C, padding=window_size//2) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_c = num / (den + eps)      # [B,C,H,W]
    ssim_mean = ssim_c.mean(dim=1, keepdim=True)
    return torch.clamp(ssim_mean, 0.0, 1.0)

def ms_ssim_map(img1: torch.Tensor, img2: torch.Tensor, scales=3):
    assert scales >= 1
    maps = []
    a = img1
    b = img2
    base_size = (img1.shape[2], img1.shape[3])
    for s in range(scales):
        m = ssim_map(a, b)
        if s > 0:
            m = F.interpolate(m, size=base_size, mode="bilinear", align_corners=False)
        maps.append(m)
        if s < scales - 1:
            a = F.avg_pool2d(a, kernel_size=2, stride=2)
            b = F.avg_pool2d(b, kernel_size=2, stride=2)
    return torch.clamp(torch.stack(maps, dim=0).mean(dim=0), 0.0, 1.0)

# ======================= LPIPS-lite (optional) ==========
class VGG16Features(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        feats = vgg.features
        self.b1 = nn.Sequential(*feats[:4])    # conv1_2
        self.b2 = nn.Sequential(*feats[4:9])   # conv2_2
        self.b3 = nn.Sequential(*feats[9:16])  # conv3_3
        self.b4 = nn.Sequential(*feats[16:23]) # conv4_3
        for p in self.parameters():
            p.requires_grad_(False)

def _vgg_get(device: torch.device, dtype: torch.dtype):
    key = (device.type, device.index if device.index is not None else -1)
    entry = _VGG_CACHE.get(key)
    if entry is not None:
        return entry["model"], entry["mean"].to(device=device, dtype=dtype), entry["std"].to(device=device, dtype=dtype)
    model = VGG16Features().to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1,3,1,1)
    _VGG_CACHE[key] = {"model": model, "mean": mean, "std": std}
    return model, mean, std

def _imagenet_norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return (x.clamp(0,1) - mean) / std

def lpips_heatmap(a: torch.Tensor, b: torch.Tensor):
    device, dtype = a.device, a.dtype
    try:
        model, mean, std = _vgg_get(device, dtype)
    except Exception as e:
        return None, f"LPIPS disabled: {str(e)}"
    with torch.no_grad():
        A = _imagenet_norm(a, mean, std)
        B = _imagenet_norm(b, mean, std)
        a1 = model.b1(A); b1 = model.b1(B)
        a2 = model.b2(a1); b2 = model.b2(b1)
        a3 = model.b3(a2); b3 = model.b3(b2)
        a4 = model.b4(a3); b4 = model.b4(b3)
        feats_a = [a1, a2, a3, a4]
        feats_b = [b1, b2, b3, b4]
        maps = []
        H, W = a.shape[2], a.shape[3]
        for fa, fb in zip(feats_a, feats_b):
            d = (fa - fb).pow(2).mean(dim=1, keepdim=True)
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
            maps.append(d)
        m = torch.stack(maps, dim=0).mean(dim=0)  # [B,1,H,W]
        m01 = normalize01(m)
        score = float(m.mean().item())
        return m01, f"LPIPS approx (VGG16 map mean): {score:.6f}"

# ======================= Color: sRGB <-> Lab (D65) =======
def _srgb_to_linear(x):
    a = 0.055
    return torch.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)

def _xyz_to_lab(xyz):
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[:,0:1,:,:] / Xn
    y = xyz[:,1:2,:,:] / Yn
    z = xyz[:,2:3,:,:] / Zn
    eps = 216/24389
    kappa = 24389/27
    def f(t):
        return torch.where(t > eps, t.pow(1/3), (kappa * t + 16) / 116)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return torch.cat([L, a, b], dim=1)

def rgb_to_lab(img_bchw):
    r = img_bchw[:,0:1,:,:]; g = img_bchw[:,1:2,:,:]; b = img_bchw[:,2:3,:,:]
    rl, gl, bl = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    X = 0.4124564*rl + 0.3575761*gl + 0.1804375*bl
    Y = 0.2126729*rl + 0.7151522*gl + 0.0721750*bl
    Z = 0.0193339*rl + 0.1191920*gl + 0.9503041*bl
    xyz = torch.cat([X, Y, Z], dim=1)
    return _xyz_to_lab(xyz)

def deltaE2000(Lab1, Lab2):
    L1, a1, b1 = Lab1[:,0:1], Lab1[:,1:2], Lab1[:,2:3]
    L2, a2, b2 = Lab2[:,0:1], Lab2[:,1:2], Lab2[:,2:3]
    kL = kC = kH = 1.0

    C1 = torch.sqrt(a1*a1 + b1*b1 + 1e-12)
    C2 = torch.sqrt(a2*a2 + b2*b2 + 1e-12)
    C_ave = 0.5*(C1 + C2)
    G = 0.5*(1 - torch.sqrt((C_ave**7) / (C_ave**7 + (25.0**7))))
    a1p = (1+G)*a1
    a2p = (1+G)*a2
    C1p = torch.sqrt(a1p*a1p + b1*b1 + 1e-12)
    C2p = torch.sqrt(a2p*a2p + b2*b2 + 1e-12)

    def atan2d(y,x):
        ang = torch.rad2deg(torch.atan2(y, x))
        ang = torch.where(ang < 0, ang + 360.0, ang)
        return ang

    h1p = atan2d(b1, a1p)
    h2p = atan2d(b2, a2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = torch.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = torch.where(dhp < -180.0, dhp + 360.0, dhp)
    dHp = 2.0*torch.sqrt(C1p*C2p + 1e-12) * torch.sin(torch.deg2rad(dhp/2.0))

    Lp_ave = 0.5*(L1 + L2)
    Cp_ave = 0.5*(C1p + C2p)

    hp_sum = h1p + h2p
    hp_diff = torch.abs(h1p - h2p)
    hp_ave = torch.where((C1p*C2p).squeeze(1) == 0, hp_sum*0.0,
                         torch.where(hp_diff <= 180.0, 0.5*hp_sum,
                                     torch.where(hp_sum < 360.0, 0.5*(hp_sum + 360.0), 0.5*(hp_sum - 360.0))))
    hp_ave = hp_ave.unsqueeze(1)

    T = (1
         - 0.17*torch.cos(torch.deg2rad(hp_ave - 30.0))
         + 0.24*torch.cos(torch.deg2rad(2*hp_ave))
         + 0.32*torch.cos(torch.deg2rad(3*hp_ave + 6.0))
         - 0.20*torch.cos(torch.deg2rad(4*hp_ave - 63.0)))

    dtheta = 30.0*torch.exp(-(((hp_ave - 275.0)/25.0)**2))
    Rc = 2.0*torch.sqrt((Cp_ave**7) / (Cp_ave**7 + (25.0**7)))
    Sl = 1 + (0.015*((Lp_ave - 50.0)**2)) / torch.sqrt(20 + ((Lp_ave - 50.0)**2))
    Sc = 1 + 0.045*Cp_ave
    Sh = 1 + 0.015*Cp_ave*T
    Rt = -torch.sin(torch.deg2rad(2.0*dtheta)) * Rc

    dE = torch.sqrt(
        (dLp/(kL*Sl))**2 +
        (dCp/(kC*Sc))**2 +
        (dHp/(kH*Sh))**2 +
        Rt * (dCp/(kC*Sc)) * (dHp/(kH*Sh)) + 1e-12
    )
    return dE  # [B,1,H,W]

# ======================= Gradients & Edge IoU ===========
def scharr_grad(gray: torch.Tensor):
    kx = torch.tensor([[3, 0, -3],
                       [10, 0, -10],
                       [3, 0, -3]], dtype=gray.dtype, device=gray.device).view(1,1,3,3) / 16.0
    ky = torch.tensor([[3, 10, 3],
                       [0,  0,  0],
                       [-3,-10,-3]], dtype=gray.dtype, device=gray.device).view(1,1,3,3) / 16.0
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-12)
    return mag, gx, gy

def percentile_threshold(x: torch.Tensor, q: float):
    """
    x: [B,1,H,W]; q in [0,1] -> per-image threshold [B,1,1,1]
    Uses torch.quantile when available, else sort-index fallback.
    """
    q = float(max(0.0, min(1.0, q)))
    B = x.shape[0]
    flat = x.view(B, -1)
    try:
        th = torch.quantile(flat, q, dim=1)  # [B]
        return th.view(B,1,1,1)
    except Exception:
        N = flat.shape[1]
        k = int(q * (N - 1))
        k = 0 if k < 0 else (N - 1 if k > (N - 1) else k)
        vals, _ = torch.sort(flat, dim=1)
        th = vals[:, k]
        return th.view(B,1,1,1)

def edge_mask(gray: torch.Tensor, q: float = 0.85):
    """
    Simple Canny-like edge mask using Gaussian blur + Scharr + percentile threshold.
    gray: [B,1,H,W] in [0,1]
    returns: [B,1,H,W] binary mask in {0,1}
    """
    blur = gaussian_blur(gray, win=5, sigma=1.0)
    mag, _, _ = scharr_grad(blur)
    th = percentile_threshold(mag, q)  # [B,1,1,1]
    mask = (mag >= th).to(gray.dtype)
    return mask

def edge_iou(mask_a: torch.Tensor, mask_b: torch.Tensor):
    inter = (mask_a * mask_b)
    union = torch.clamp(mask_a + mask_b, 0, 1)
    iou = (inter.sum(dim=(2,3), keepdim=True) + 1e-6) / (union.sum(dim=(2,3), keepdim=True) + 1e-6)
    vis = torch.zeros_like(mask_a)
    vis = torch.where(inter > 0, torch.ones_like(vis), vis)
    vis = torch.where((union>0) & (inter==0), torch.full_like(vis, 0.3), vis)
    return vis.clamp(0,1), iou  # vis [B,1,H,W], scalar IoU per-image

# ======================= Wavelet (Haar) ==================
def haar_dwt2(x):
    h = torch.tensor([1.0, 1.0], device=x.device, dtype=x.dtype).view(1,1,1,2) / math.sqrt(2.0)
    g = torch.tensor([1.0,-1.0], device=x.device, dtype=x.dtype).view(1,1,1,2) / math.sqrt(2.0)
    C = x.shape[1]
    h = h.expand(C,1,1,2); g = g.expand(C,1,1,2)
    low = F.conv2d(x, h, groups=C, stride=(1,2), padding=(0,0))
    high= F.conv2d(x, g, groups=C, stride=(1,2), padding=(0,0))
    hT = h.transpose(2,3)
    gT = g.transpose(2,3)
    LL = F.conv2d(low,  hT, groups=C, stride=(2,1), padding=(0,0))
    LH = F.conv2d(low,  gT, groups=C, stride=(2,1), padding=(0,0))
    HL = F.conv2d(high, hT, groups=C, stride=(2,1), padding=(0,0))
    HH = F.conv2d(high, gT, groups=C, stride=(2,1), padding=(0,0))
    return LL, LH, HL, HH

def wavelet_band_diff(a, b, levels=3):
    baseH, baseW = a.shape[2], a.shape[3]
    accum = None
    A = a
    B = b
    for _ in range(levels):
        LA, LHa, HLa, HHa = haar_dwt2(A)
        LB, LHb, HLb, HHb = haar_dwt2(B)
        diff = (torch.abs(LHa - LHb) + torch.abs(HLa - HLb) + torch.abs(HHa - HHb)).mean(dim=1, keepdim=True)
        diff = F.interpolate(diff, size=(baseH, baseW), mode="bilinear", align_corners=False)
        accum = diff if accum is None else accum + diff
        A, B = LA, LB
    return normalize01(accum)

# ======================= Denoiser Residuum ===============
def denoiser_residual(img: torch.Tensor, win=11, sigma=2.0, gain=2.0):
    low = gaussian_blur(img, win=win, sigma=sigma)
    res = img - low
    vis = torch.clamp(0.5 + gain * res, 0.0, 1.0)
    return vis

# ======================= FFT / Spectra ===================
def fft_magnitude(x_gray: torch.Tensor, log_eps: float = 1e-3):
    x = x_gray - x_gray.mean(dim=(2,3), keepdim=True)
    fx = torch.fft.fft2(x.float())
    fx = torch.fft.fftshift(fx, dim=(-2,-1))
    mag = torch.log(torch.abs(fx) + log_eps)
    return normalize01(mag)

# ======================= Round-trip ======================
def resample(img: torch.Tensor, scale: float, kernel: str = "bicubic"):
    H, W = img.shape[2], img.shape[3]
    newH = max(1, int(round(H * scale)))
    newW = max(1, int(round(W * scale)))
    mode = "bicubic" if kernel == "bicubic" else ("bilinear" if kernel == "bilinear" else "area")
    down = F.interpolate(img, size=(newH,newW), mode=mode, align_corners=False if mode!="area" else None)
    up   = F.interpolate(down, size=(H,W), mode=mode, align_corners=False if mode!="area" else None)
    return up

# ======================= Node ============================
class MultiAnalysisNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "align_mode": (["resize_b_to_a", "center_crop_to_min", "pad_to_max"],),
                "color_map": (["jet", "grayscale"],),
                "invert_ssim": ("BOOLEAN", {"default": True}),
                "ms_scales": ("INT", {"default": 3, "min": 1, "max": 5}),
                "enable_lpips": ("BOOLEAN", {"default": False}),
                # Edge settings
                "edge_percentile": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 0.99, "step": 0.01}),
                # Wavelet settings
                "wavelet_levels": ("INT", {"default": 3, "min": 1, "max": 4}),
                # Denoiser
                "denoise_sigma": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 5.0, "step": 0.1}),
                "denoise_gain": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                # Round-trip
                "roundtrip_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 0.95, "step": 0.05}),
                "roundtrip_kernel": (["bicubic", "bilinear", "area"],),
            }
        }

    RETURN_TYPES = (
        # Core
        "IMAGE",  # ssim_heatmap
        "IMAGE",  # dssim_heatmap
        "IMAGE",  # ms_ssim_heatmap
        "IMAGE",  # lpips_heatmap
        "IMAGE",  # diff_abs_heatmap
        "IMAGE",  # residual_visual
        "IMAGE",  # fft_orig
        "IMAGE",  # fft_upscaled
        "IMAGE",  # fft_diff
        "STRING", # ssim_text
        "STRING", # lpips_text
        # Color & ΔE2000
        "IMAGE",  # deltaE2000_heatmap
        "IMAGE",  # L_diff_heatmap
        "IMAGE",  # a_diff_heatmap
        "IMAGE",  # b_diff_heatmap
        # Gradients & edges
        "IMAGE",  # gradient_diff_heatmap
        "IMAGE",  # edge_iou_heatmap
        "STRING", # edge_iou_text
        # Wavelet
        "IMAGE",  # wavelet_band_diff_heatmap
        # Denoiser residuum
        "IMAGE",  # denoiser_residual_visual
        # Round-trip
        "IMAGE",  # roundtrip_diff_heatmap
    )
    RETURN_NAMES = (
        "ssim_heatmap",
        "dssim_heatmap",
        "ms_ssim_heatmap",
        "lpips_heatmap",
        "diff_abs_heatmap",
        "residual_visual",
        "fft_orig",
        "fft_upscaled",
        "fft_diff",
        "ssim_text",
        "lpips_text",
        "deltaE2000_heatmap",
        "L_diff_heatmap",
        "a_diff_heatmap",
        "b_diff_heatmap",
        "gradient_diff_heatmap",
        "edge_iou_heatmap",
        "edge_iou_text",
        "wavelet_band_diff_heatmap",
        "denoiser_residual_visual",
        "roundtrip_diff_heatmap",
    )
    FUNCTION = "analyze"
    CATEGORY = NODE_CATEGORY

    def analyze(self, image_a, image_b, align_mode, color_map, invert_ssim, ms_scales, enable_lpips,
                edge_percentile, wavelet_levels, denoise_sigma, denoise_gain, roundtrip_scale, roundtrip_kernel):

        # Device/Dtype
        image_a, image_b = ensure_same_device(image_a, image_b)
        a = to_bchw(image_a)
        b = to_bchw(image_b)
        # Channels
        if a.shape[1] == 1: a = a.repeat(1,3,1,1)
        if b.shape[1] == 1: b = b.repeat(1,3,1,1)
        # Numerics
        a = a.to(dtype=torch.float32, copy=False)
        b = b.to(dtype=torch.float32, copy=False)

        device_type = "cuda" if a.device.type == "cuda" else "cpu"
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            # Align
            if align_mode == "resize_b_to_a":
                b = resize_like(b, a)
            elif align_mode == "center_crop_to_min":
                a, b = center_crop_to_min(a, b)
            else:
                a, b = pad_to_max(a, b)

            # ---------- SSIM & DSSIM ----------
            ssim = ssim_map(a, b)                        # [B,1,H,W]
            dssim = 1.0 - ssim
            ssim_vis = dssim if invert_ssim else ssim
            ssim_heat  = apply_colormap(normalize01(ssim_vis), color_map)
            dssim_heat = apply_colormap(normalize01(dssim), color_map)
            ssim_text = f"SSIM mean: {float(ssim.mean().item()):.6f}"

            # ---------- MS-SSIM ----------
            ms = ms_ssim_map(a, b, scales=int(ms_scales))
            ms_vis = (1.0 - ms) if invert_ssim else ms
            ms_heat = apply_colormap(normalize01(ms_vis), color_map)

            # ---------- LPIPS (optional) ----------
            if enable_lpips:
                lp_map, lp_text = lpips_heatmap(a, b)
                lpips_heat = apply_colormap(lp_map, color_map) if lp_map is not None else torch.zeros_like(ssim_heat)
            else:
                lpips_heat = torch.zeros_like(ssim_heat)
                lp_text = "LPIPS disabled"

            # ---------- Per-pixel abs diff ----------
            diff_abs = torch.abs(a - b).mean(dim=1, keepdim=True)
            diff_abs_heat = apply_colormap(normalize01(diff_abs), color_map)

            # ---------- Residual (signed, quick) ----------
            resid_vis = torch.clamp(0.5 + (b - a)*2.0, 0.0, 1.0)

            # ---------- FFT spectra ----------
            a_gray = (0.2989*a[:,0:1] + 0.5870*a[:,1:2] + 0.1140*a[:,2:3]).clamp_(0,1)
            b_gray = (0.2989*b[:,0:1] + 0.5870*b[:,1:2] + 0.1140*b[:,2:3]).clamp_(0,1)
            fa = fft_magnitude(a_gray, 1e-3)
            fb = fft_magnitude(b_gray, 1e-3)
            f_diff = normalize01(torch.abs(fa - fb))
            fft_a = apply_colormap(fa, color_map)
            fft_b = apply_colormap(fb, color_map)
            fft_d = apply_colormap(f_diff, color_map)

            # ---------- ΔE2000 + Lab diffs ----------
            LabA = rgb_to_lab(a.clamp(0,1))
            LabB = rgb_to_lab(b.clamp(0,1))
            dE = deltaE2000(LabA, LabB)                      # [B,1,H,W]
            dE_heat = apply_colormap(normalize01(dE), color_map)
            Ldiff = normalize01(torch.abs(LabA[:,0:1] - LabB[:,0:1]))
            adiff = normalize01(torch.abs(LabA[:,1:2] - LabB[:,1:2]))
            bdiff = normalize01(torch.abs(LabA[:,2:3] - LabB[:,2:3]))
            Ldiff_heat = apply_colormap(Ldiff, color_map)
            adiff_heat = apply_colormap(adiff, color_map)
            bdiff_heat = apply_colormap(bdiff, color_map)

            # ---------- Gradients & Edge IoU ----------
            gA, _, _ = scharr_grad(a_gray)
            gB, _, _ = scharr_grad(b_gray)
            gdiff = normalize01(torch.abs(gA - gB))
            gdiff_heat = apply_colormap(gdiff, color_map)

            mA = edge_mask(a_gray, q=float(edge_percentile))
            mB = edge_mask(b_gray, q=float(edge_percentile))
            edge_vis, iou_scalar = edge_iou(mA, mB)
            edge_heat = apply_colormap(edge_vis, color_map)
            edge_text = f"Edge IoU (percentile {edge_percentile:.2f}): {float(iou_scalar.mean().item()):.6f}"

            # ---------- Wavelet band diff ----------
            wav = wavelet_band_diff(a, b, levels=int(wavelet_levels))
            wav_heat = apply_colormap(wav, color_map)

            # ---------- Denoiser residuum ----------
            den_vis = denoiser_residual(b, win=11, sigma=float(denoise_sigma), gain=float(denoise_gain))

            # ---------- Round-trip stability ----------
            rt = resample(b, scale=float(roundtrip_scale), kernel=str(roundtrip_kernel))
            rt_diff = normalize01(torch.abs(b - rt).mean(dim=1, keepdim=True))
            rt_heat = apply_colormap(rt_diff, color_map)

        # BHWC (3ch) & return
        return (
            _ensure_bhwc3(ssim_heat),
            _ensure_bhwc3(dssim_heat),
            _ensure_bhwc3(ms_heat),
            _ensure_bhwc3(lpips_heat),
            _ensure_bhwc3(diff_abs_heat),
            _ensure_bhwc3(resid_vis),
            _ensure_bhwc3(fft_a),
            _ensure_bhwc3(fft_b),
            _ensure_bhwc3(fft_d),
            ssim_text,
            lp_text,
            _ensure_bhwc3(dE_heat),
            _ensure_bhwc3(Ldiff_heat),
            _ensure_bhwc3(adiff_heat),
            _ensure_bhwc3(bdiff_heat),
            _ensure_bhwc3(gdiff_heat),
            _ensure_bhwc3(edge_heat),
            edge_text,
            _ensure_bhwc3(wav_heat),
            _ensure_bhwc3(den_vis),
            _ensure_bhwc3(rt_heat),
        )

# ======================= Registration ====================
NODE_CLASS_MAPPINGS = {
    "MultiAnalysisNode": MultiAnalysisNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiAnalysisNode": NODE_NAME
}
