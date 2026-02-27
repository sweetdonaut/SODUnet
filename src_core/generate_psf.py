"""
PSF Defect Generator
====================
Annular aperture + point source -> FFT -> noise -> PSF -> connected peak cleaning

Usage:
  python generate_psf.py --config defects/type1.yaml
"""

import argparse
import numpy as np
import os
import yaml
from scipy.ndimage import label


PARAM_NAMES = [
    "outer_r", "epsilon", "ellipticity", "ellip_angle",
    "defocus", "astig_x", "astig_y", "coma_x", "coma_y",
    "spherical", "trefoil_x", "trefoil_y",
    "brightness", "background", "gaussian_sigma",
]

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in cfg:
        val = cfg[key]
        if isinstance(val, list) and len(val) == 2 and all(isinstance(v, (int, float)) for v in val):
            cfg[key] = tuple(val)
    return cfg


def sample(rng, r):
    if isinstance(r, (list, tuple)):
        return r[0] if r[0] == r[1] else rng.uniform(r[0], r[1])
    return r


def generate_one(cfg, rng):
    N = cfg["psf_size"]
    y, x = np.mgrid[-N//2:N//2, -N//2:N//2].astype(np.float64)

    outer_r = sample(rng, cfg["outer_r"])
    eps = sample(rng, cfg["epsilon"])
    ellip = sample(rng, cfg["ellipticity"])
    ellip_ang = sample(rng, cfg["ellip_angle"])
    defocus = sample(rng, cfg["defocus"])
    astig_x = sample(rng, cfg["astig_x"])
    astig_y = sample(rng, cfg["astig_y"])
    coma_x = sample(rng, cfg["coma_x"])
    coma_y = sample(rng, cfg["coma_y"])
    sph = sample(rng, cfg["spherical"])
    tri_x = sample(rng, cfg["trefoil_x"])
    tri_y = sample(rng, cfg["trefoil_y"])
    brightness = sample(rng, cfg["brightness"])
    bg = sample(rng, cfg["background"])
    g_sig = sample(rng, cfg["gaussian_sigma"])

    # Annular mask
    cos_a, sin_a = np.cos(np.radians(ellip_ang)), np.sin(np.radians(ellip_ang))
    rx = (x * cos_a + y * sin_a) / (1 + ellip)
    ry = (-x * sin_a + y * cos_a) / (1 - ellip)
    r = np.sqrt(rx**2 + ry**2)
    mask = ((r <= outer_r) & (r >= outer_r * eps)).astype(np.float64)

    # Phase (Zernike)
    dx, dy = x / outer_r, y / outer_r
    rho2 = dx**2 + dy**2
    rho = np.sqrt(rho2)
    theta = np.arctan2(dy, dx)
    phase = (defocus * (2*rho2 - 1)
             + astig_x * rho2 * np.cos(2*theta)
             + astig_y * rho2 * np.sin(2*theta)
             + coma_x * (3*rho2 - 2) * rho * np.cos(theta)
             + coma_y * (3*rho2 - 2) * rho * np.sin(theta)
             + sph * (6*rho2**2 - 6*rho2 + 1)
             + tri_x * rho2 * rho * np.cos(3*theta)
             + tri_y * rho2 * rho * np.sin(3*theta))

    # PSF = |FFT(pupil)|^2
    psf = np.abs(np.fft.fftshift(np.fft.fft2(mask * np.exp(1j * phase))))**2

    # Brightness + background + noise
    psf = psf / psf.sum() * brightness + bg
    if cfg.get("poisson_noise", True):
        psf = rng.poisson(np.maximum(0, psf)).astype(np.float64)
    if g_sig > 0:
        psf += rng.normal(0, g_sig, psf.shape)
    psf = np.maximum(0, psf)

    # Center crop
    c = cfg["crop_size"]
    s = N // 2 - c // 2
    cropped = psf[s:s+c, s:s+c]

    params = [outer_r, eps, ellip, ellip_ang,
              defocus, astig_x, astig_y, coma_x, coma_y, sph, tri_x, tri_y,
              brightness, bg, g_sig]
    return cropped, params


def clean_connected_peak(img, threshold_multiplier=1.0):
    """Keep only the connected region around the brightest pixel."""
    mu, sigma = img.mean(), img.std()
    bg = np.median(img)
    peak_y, peak_x = np.unravel_index(img.argmax(), img.shape)

    binary = img > (mu + threshold_multiplier * sigma)
    labeled, _ = label(binary)
    peak_label = labeled[peak_y, peak_x]

    if peak_label == 0:
        return np.zeros_like(img)

    core_mask = (labeled == peak_label)
    cleaned = np.where(core_mask, img - bg, 0)
    return np.maximum(cleaned, 0).astype(np.float32)


def create_psf_defect(cfg):
    """Generate one PSF defect on the fly, cleaned and cropped to bounding box.
    Returns normalized 0-1 float32 array, or None if generation fails.
    """
    rng = np.random.default_rng()
    raw, _ = generate_one(cfg, rng)
    thr = cfg.get("threshold_multiplier", 1.0)
    cleaned = clean_connected_peak(raw, thr)

    nonzero = np.argwhere(cleaned > 0)
    if len(nonzero) == 0:
        return None

    y_min, x_min = nonzero.min(axis=0)
    y_max, x_max = nonzero.max(axis=0)
    cropped = cleaned[y_min:y_max+1, x_min:x_max+1]

    if cropped.max() > 0:
        cropped = cropped / cropped.max()

    return cropped.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    n = cfg["n_samples"]
    c = cfg["crop_size"]
    thr_mult = cfg.get("threshold_multiplier", 1.0)
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)

    rng = np.random.default_rng(42)
    images = np.zeros((n, c, c), dtype=np.float32)
    params = np.zeros((n, len(PARAM_NAMES)), dtype=np.float32)

    print(f"Generating {n} PSFs (threshold_multiplier={thr_mult}) ...")
    for i in range(n):
        raw, params[i] = generate_one(cfg, rng)
        images[i] = clean_connected_peak(raw, thr_mult)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}")

    np.save(f"{out}/psf_images.npy", images)
    np.save(f"{out}/psf_params.npy", params)
    with open(f"{out}/params_names.txt", "w") as f:
        f.write("\n".join(PARAM_NAMES))

    nonzero_counts = [(img > 0).sum() for img in images]
    print(f"\nDone!")
    print(f"  images: {out}/psf_images.npy  {images.shape}")
    print(f"  params: {out}/psf_params.npy  {params.shape}")
    print(f"  core size: mean={np.mean(nonzero_counts):.1f}px, range=[{np.min(nonzero_counts)}, {np.max(nonzero_counts)}]px")


if __name__ == "__main__":
    main()
