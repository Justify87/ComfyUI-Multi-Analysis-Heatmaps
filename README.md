# ComfyUI Multi-Analysis Heatmaps

A custom ComfyUI node for **visual comparison of two images** using multiple perceptual and mathematical methods.  
The goal: make hidden differences *visible* as colorful heatmaps, so you can **see** where an upscaler, denoiser, or diffusion model changed your image — even when your eyes can’t tell at first glance.

---

## ✨ Features

Each method outputs its own heatmap (and sometimes a numeric score) so you can inspect everything side by side:

- **SSIM / DSSIM / MS-SSIM**  
  Structural similarity.  
  - Bright = dissimilar, dark = similar.  
  - DSSIM is the inverse (bright = more difference).  
  - MS-SSIM averages across multiple scales (robust to small shifts).

- **LPIPS (VGG16, perceptual map)**  
  Estimates whether differences are **visible to human perception**.  
  - Bright areas = regions that look different to the eye.  
  - Dark = visually similar.

- **Per-Pixel Difference (Abs)**  
  Direct pixel math.  
  - Bright = strong raw value difference, even if imperceptible.

- **Residual (Signed)**  
  Highlights what B has **added or removed** compared to A.  
  - Bright = added details/noise.  
  - Dark = lost details.

- **FFT / Spectrum Analysis**  
  Looks at **frequency content** (textures, fine detail).  
  - Shows original spectrum, upscaled spectrum, and difference.  
  - Large changes in high frequencies = sharpening or hallucinated texture.

- **ΔE2000 (Color Difference)**  
  Shows perceptual color shifts in Lab color space.  
  - Includes total ΔE2000 heatmap and individual **L\*** (lightness), **a\*** (red/green), **b\*** (blue/yellow) differences.

- **Gradient Difference (Scharr)**  
  Compares edge strength.  
  - Bright = edge got stronger/weaker.  
  - Useful to detect oversharpening.

- **Edge IoU (Canny-like)**  
  Compares binary edge maps of A and B.  
  - Bright = edges overlap.  
  - Gray = edges differ.  
  - Includes IoU score as text output.

- **Wavelet Band Difference**  
  Multi-scale Haar wavelet decomposition.  
  - Bright = changes in fine details/texture.

- **Denoiser Residual**  
  Highlights **high-frequency noise patterns**.  
  - Bright = new noise/texture in B.  
  - Can reveal “fake texture” added by diffusion upscalers.

- **Round-Trip Stability**  
  Downscale & re-upscale B, compare to itself.  
  - Bright = unstable areas (info lost or artifacts added).  
  - Useful to judge robustness, not just looks.

---

## 🧑‍🎨 How to Interpret

Think of each heatmap as a **different pair of glasses**:

- **Dark = good match** (few differences)  
- **Bright = more change** (differences, artifacts, or enhancements)  

Use the right “glasses” depending on what you care about:
- **Overall similarity:** SSIM / DSSIM  
- **Human perception:** LPIPS, MS-SSIM  
- **Color shifts:** ΔE2000, a\*, b\*  
- **Sharpness / edges:** Gradient Diff, Edge IoU  
- **Textures / hallucinations:** FFT Diff, Wavelet Diff, Denoiser Residual  
- **Robustness:** Round-Trip Stability  

---

## ⚠️ Disclaimer

This project is **vibe-coded**:  
- Heatmaps are **heuristic visualizations**.  
- Bright ≠ bad, dark ≠ good — interpretation depends on context.  
- The node does *not* decide quality for you.  

👉 **Interpretation and responsibility are fully on the user.**  
These tools are meant for exploration and intuition, not absolute ground-truth judgment.

---

## 📸 Example

![Heatmap Example](Screenshot.png)

---

## 🛠 Installation

1. Clone or download this repo into your `ComfyUI/custom_nodes/` folder:
   ```bash
   git clone https://github.com/yourname/ComfyUI-MultiAnalysisHeatmaps


## Additionals

- Upscaler Used in the example workflow:
- https://github.com/daredevilstudio/AI-upscaling-models
- https://github.com/daredevilstudio/AI-upscaling-models/releases/download/PureVision/2x_PureVision.pth
- image compare node:
- https://github.com/quasiblob/ComfyUI-EsesImageCompare
