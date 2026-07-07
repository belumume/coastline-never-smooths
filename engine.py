"""Fractal-trainability core engine (v1, oracle-aligned).

Faithful torch port of Sohl-Dickstein's reference implementation
(github.com/Sohl-Dickstein/fractal, MIT), validated cell-by-cell against
reference_code.py extracted from the author's notebook:

- mean-field parameterization: Z = X @ W / sqrt(fan_in); tanh(sqrt(2)*Z);
  final output = Z_last / sqrt(width). Weights init ~ N(0,1) unscaled.
- dataset: n_data = n_params / target_dim (dataset_param_multiple = 1),
  X ~ N(0,1) [n_data, width], Y ~ N(0,1) [n_data, 1].
- axes: per-layer learning rates (lr input layer vs lr output layer),
  log10 window canonical start = [-3, 6] x [-3, 6].
- readout: signed convergence measure, reference cell 8 —
  v normalized by v[0], clipped at 1e6; converged iff mean(v[-20:]) < 1;
  score = -sum(v) if converged else sum(1/v).
- rendering: rank-based cdf_img normalization to [-1, 1], Spectral cmap.
- fractal dimension: sign-change edges + box counting (reference cell 10,
  porespy replaced by a dependency-free boxcount below).

Known deviations (deliberate): torch instead of JAX (molab preinstalls
torch); fp32 default with fp64 opt-in for deep zooms; different RNG draws
(structural fidelity, not pixel-identical images).

NOVELTY NOTE for the notebook: the author's own minibatch run (cell 27)
uses ONE fixed minibatch schedule shared by every grid point, so his
boundary remains a deterministic set. Our extension treats the boundary
as an ensemble object: P(converged) over K independent schedules, and
fractal dimension of the probabilistic contour vs batch size.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch


@dataclasses.dataclass
class Config:
    width: int = 16
    target_dim: int = 1
    dataset_param_multiple: float = 1.0
    steps: int = 500
    max_val: float = 1e6
    tail: int = 20  # steps averaged for the converged test
    seed: int = 0
    dtype: torch.dtype = torch.float32

    @property
    def n_params(self) -> int:
        return self.width * self.width + self.width * self.target_dim

    @property
    def n_data(self) -> int:
        return int(self.n_params * self.dataset_param_multiple / self.target_dim)


def make_problem(cfg: Config, device: torch.device):
    """Fixed dataset + fixed unit-normal init, shared across all grid points."""
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    X = torch.randn(cfg.n_data, cfg.width, generator=g, dtype=cfg.dtype)
    Y = torch.randn(cfg.n_data, cfg.target_dim, generator=g, dtype=cfg.dtype)
    W1 = torch.randn(cfg.width, cfg.width, generator=g, dtype=cfg.dtype)
    W2 = torch.randn(cfg.width, cfg.target_dim, generator=g, dtype=cfg.dtype)
    return (t.to(device) for t in (X, Y, W1, W2))


def _forward_and_grads(W1, W2, Xb, Yb, width: int):
    """Mean-field 1-hidden-layer net, manual closed-form backprop.

    forward (reference cell 7):
      Z1 = Xb @ W1 / sqrt(width);  H = tanh(sqrt(2) * Z1)
      Z2 = H @ W2 / sqrt(width);   out = Z2 / sqrt(width)
      loss = mean((out - Yb)^2)
    Shapes: W1 [G,D,H], W2 [G,H,1], Xb [N,D], Yb [N,1].
    """
    s = width**-0.5
    Z1 = torch.einsum("nd,gdh->gnh", Xb, W1) * s
    H = torch.tanh(Z1 * (2.0**0.5))
    Z2 = torch.einsum("gnh,gho->gno", H, W2) * s
    out = Z2 * s
    err = out - Yb.unsqueeze(0)  # [G,N,1]
    n_el = err.shape[1] * err.shape[2]

    gout = (2.0 / n_el) * err
    gZ2 = gout * s
    gW2 = torch.einsum("gnh,gno->gho", H, gZ2) * s
    gH = torch.einsum("gno,gho->gnh", gZ2, W2) * s
    gZ1 = gH * (1.0 - H**2) * (2.0**0.5)
    gW1 = torch.einsum("nd,gnh->gdh", Xb, gZ1) * s

    loss = (err**2).mean(dim=(1, 2))
    return loss, gW1, gW2


@torch.no_grad()
def _full_loss(W1, W2, X, Y, width: int):
    s = width**-0.5
    H = torch.tanh(torch.einsum("nd,gdh->gnh", X, W1) * s * (2.0**0.5))
    out = torch.einsum("gnh,gho->gno", H, W2) * s * s
    return ((out - Y.unsqueeze(0)) ** 2).mean(dim=(1, 2))


def convergence_grid(
    cfg: Config,
    lrs1: torch.Tensor,
    lrs2: torch.Tensor,
    device: torch.device,
    batch_size: int | None = None,
    noise_seed: int = 42,
) -> torch.Tensor:
    """Signed convergence measure per (lr1, lr2) pair -> [G].

    Faithful to reference cells 7-8. batch_size=None => full batch
    (paper's deterministic setting). batch_size=k => minibatch indices
    drawn with replacement each step from generator seeded noise_seed
    (one schedule shared across the grid, matching reference cell 27;
    the ensemble extension varies noise_seed).
    """
    X, Y, W1_0, W2_0 = make_problem(cfg, device)
    G = lrs1.shape[0]
    W1 = W1_0.unsqueeze(0).repeat(G, 1, 1)
    W2 = W2_0.unsqueeze(0).repeat(G, 1, 1)
    lr1 = lrs1.view(G, 1, 1).to(device=device, dtype=cfg.dtype)
    lr2 = lrs2.view(G, 1, 1).to(device=device, dtype=cfg.dtype)

    gnoise = torch.Generator(device="cpu").manual_seed(noise_seed)
    v_sum = torch.zeros(G, dtype=cfg.dtype, device=device)  # sum of v
    inv_sum = torch.zeros(G, dtype=cfg.dtype, device=device)  # sum of 1/v
    tail_sum = torch.zeros(G, dtype=cfg.dtype, device=device)
    # A run that ever goes non-finite is DEAD: its tracked value stays pinned
    # at max_val forever (reference cell 8 semantics — params are never
    # revived). We zero dead params only to stop NaN churn in the compute;
    # the dead mask keeps the *measure* honest.
    dead = torch.zeros(G, dtype=torch.bool, device=device)
    v0 = None

    for step in range(cfg.steps):
        if batch_size is None:
            Xb, Yb = X, Y
        else:
            idx = torch.randint(0, cfg.n_data, (batch_size,), generator=gnoise)
            Xb, Yb = X[idx.to(device)], Y[idx.to(device)]

        with torch.no_grad():
            # loss tracked on the FULL batch (reference cell 7 comment)
            loss_full = _full_loss(W1, W2, X, Y, cfg.width)
            dead |= ~torch.isfinite(loss_full)
            lossb, gW1, gW2 = _forward_and_grads(W1, W2, Xb, Yb, cfg.width)
            W1 = W1 - lr1 * gW1
            W2 = W2 - lr2 * gW2
            if dead.any():
                W1[dead] = 0.0
                W2[dead] = 0.0

            v = torch.where(dead, torch.full_like(loss_full, cfg.max_val), loss_full)
            if v0 is None:
                v0 = v.clamp_min(1e-30)
            v = (v / v0).clamp_max(cfg.max_val)
            v = torch.where(dead, torch.full_like(v, cfg.max_val), v)
            v_sum += v
            inv_sum += 1.0 / v.clamp_min(1e-30)
            if step >= cfg.steps - cfg.tail:
                tail_sum += v

    converged = (tail_sum / cfg.tail) < 1.0
    return torch.where(converged, -v_sum, inv_sum)


def compute_grid(
    cfg: Config,
    res: int,
    device: str | torch.device = "cpu",
    window: tuple[float, float, float, float] = (-3.0, 6.0, -3.0, 6.0),
    batch_size: int | None = None,
    noise_seed: int = 42,
    chunk: int = 16384,
) -> np.ndarray:
    """Signed-convergence image [res, res]; window in log10 lr, (x=lr1, y=lr2)."""
    device = torch.device(device)
    a0, a1, b0, b1 = window
    lr1 = torch.logspace(a0, a1, res, dtype=cfg.dtype)
    lr2 = torch.logspace(b0, b1, res, dtype=cfg.dtype)
    L1, L2 = torch.meshgrid(lr1, lr2, indexing="xy")
    flat1, flat2 = L1.reshape(-1), L2.reshape(-1)

    out = torch.empty(res * res, dtype=cfg.dtype)
    for i in range(0, res * res, chunk):
        j = min(i + chunk, res * res)
        out[i:j] = convergence_grid(
            cfg, flat1[i:j], flat2[i:j], device, batch_size, noise_seed
        ).cpu()
    return out.reshape(res, res).numpy()


def cdf_img(
    x: np.ndarray, x_ref: np.ndarray | None = None, buffer: float = 0.25
) -> np.ndarray:
    """Rank-normalize to [-1, 1] preserving sign (reference cell 11)."""
    if x_ref is None:
        x_ref = x
    u = np.sort(x_ref.ravel())
    num_neg = int((u < 0).sum())
    num_nonneg = u.size - num_neg
    v = np.concatenate(
        [np.linspace(-1, -buffer, num_neg), np.linspace(buffer, 1, num_nonneg)]
    )
    return -np.interp(x, u, v)


def extract_edges(img: np.ndarray) -> np.ndarray:
    """Boundary pixels = sign changes among 2x2 neighborhoods (reference cell 10)."""
    y = np.stack([img[1:, 1:], img[:-1, 1:], img[1:, :-1], img[:-1, :-1]], axis=-1)
    return (y.max(axis=-1) * y.min(axis=-1)) < 0


def boxcount_dimension(edges: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Box-counting dimension of a binary edge image (porespy-free).

    Returns (median local slope, box sizes, counts) — same estimator shape
    as the reference's ps.metrics.boxcount + median-of-slopes.
    """
    n = min(edges.shape)
    sizes = []
    counts = []
    k = 1
    while k <= n // 2:
        h = (edges.shape[0] // k) * k
        w = (edges.shape[1] // k) * k
        trimmed = edges[:h, :w]
        blocks = trimmed.reshape(h // k, k, w // k, k).any(axis=(1, 3))
        sizes.append(k)
        counts.append(int(blocks.sum()))
        k *= 2
    sizes = np.array(sizes, dtype=float)
    counts = np.array(counts, dtype=float)
    valid = counts > 0
    log_s, log_c = np.log(sizes[valid]), np.log(counts[valid])
    slopes = -(np.diff(log_c) / np.diff(log_s))
    return float(np.median(slopes)), sizes, counts


def loss_trace(
    cfg: Config,
    lr1: float,
    lr2: float,
    device: str | torch.device = "cpu",
    batch_size: int | None = None,
    noise_seed: int = 42,
) -> tuple[np.ndarray, bool]:
    """Tracked value v at every step for ONE (lr1, lr2) point.

    Identical dynamics to convergence_grid with G=1 (same problem, same
    init, same update, same dead-run semantics), recording v each step so
    a single pixel can be unfolded back into its full training run.
    Returns (v trace [steps], converged flag). v1.3 addition for the
    notebook's crosshair replay panel.
    """
    device = torch.device(device)
    X, Y, W1_0, W2_0 = make_problem(cfg, device)
    W1 = W1_0.unsqueeze(0).clone()
    W2 = W2_0.unsqueeze(0).clone()
    l1 = torch.tensor(lr1, dtype=cfg.dtype, device=device).view(1, 1, 1)
    l2 = torch.tensor(lr2, dtype=cfg.dtype, device=device).view(1, 1, 1)
    gnoise = torch.Generator(device="cpu").manual_seed(noise_seed)
    dead = torch.zeros(1, dtype=torch.bool, device=device)
    v0 = None
    vs = []
    for step in range(cfg.steps):
        if batch_size is None:
            Xb, Yb = X, Y
        else:
            idx = torch.randint(0, cfg.n_data, (batch_size,), generator=gnoise)
            Xb, Yb = X[idx.to(device)], Y[idx.to(device)]
        with torch.no_grad():
            loss_full = _full_loss(W1, W2, X, Y, cfg.width)
            dead |= ~torch.isfinite(loss_full)
            lossb, gW1, gW2 = _forward_and_grads(W1, W2, Xb, Yb, cfg.width)
            W1 = W1 - l1 * gW1
            W2 = W2 - l2 * gW2
            if dead.any():
                W1[dead] = 0.0
                W2[dead] = 0.0
            v = torch.where(dead, torch.full_like(loss_full, cfg.max_val), loss_full)
            if v0 is None:
                v0 = v.clamp_min(1e-30)
            v = (v / v0).clamp_max(cfg.max_val)
            v = torch.where(dead, torch.full_like(v, cfg.max_val), v)
            vs.append(v)
    trace = torch.cat(vs).float().cpu().numpy()
    converged = bool(trace[-cfg.tail :].mean() < 1.0)
    return trace, converged


def main():
    cfg = Config(steps=500)
    res = 512
    t0 = time.time()
    img = compute_grid(cfg, res=res, device="cpu")
    dt = time.time() - t0
    frac_conv = float((img < 0).mean())
    dim, _, _ = boxcount_dimension(extract_edges(img))
    print(
        f"grid {res}x{res} x {cfg.steps} steps in {dt:.0f}s | "
        f"converged frac {frac_conv:.3f} | boundary dim {dim:.3f}"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=130)
    ax.imshow(
        cdf_img(img),
        origin="lower",
        cmap="Spectral",
        extent=[-3, 6, -3, 6],
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel("log10 lr (input layer)")
    ax.set_ylabel("log10 lr (output layer)")
    ax.set_title("Trainability, mean-field tanh (paper-canonical view)")
    fig.tight_layout()
    fig.savefig("fractal_v1_canonical.png")
    np.save("fractal_v1_canonical.npy", img)
    print("wrote fractal_v1_canonical.png / .npy")


if __name__ == "__main__":
    main()
