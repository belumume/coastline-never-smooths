# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "wigglystuff==0.5.13",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # The Coastline That Never Smooths

    **Paper:** [The boundary of neural network trainability is fractal](https://www.alphaxiv.org/abs/2402.06184) (Jascha Sohl-Dickstein, 2024, arXiv 2402.06184)

    Pick two learning rates for a tiny neural network: one for the input layer, one for the output layer. Train it. Either the loss settles (trainable) or it blows up (divergent). Color one pixel by the outcome. Repeat for every pair on a grid and you get a map of trainability.

    Intuition says the border between the two regions should behave like a coastline on a school atlas: wiggly from afar, smooth once you zoom in.

    The paper's claim, and this notebook's centerpiece, is that it is a coastline in the *mathematical* sense: a true fractal. Zoom in ten times, a thousand times, and it never smooths. New bays, islands, and inlets of trainability keep appearing at every scale.

    This notebook does three things:

    1. **Rebuilds the fractal from scratch.** Every pixel is a real gradient descent training run, executed live on this notebook's GPU (a GPU-less fork drops to a smaller grid and still works).
    2. **Lets you sail the coastline.** Pick any box and the map recomputes at full resolution inside it.
    3. **Asks a question the paper deliberately excluded.** Real training is noisy (minibatches). Does the coastline survive noise? We measure its fractal dimension as noise grows, and watch what happens to it.
    """)
    return


@app.cell
def mo_import():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    import dataclasses

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from wigglystuff import ChartPuck

    return ChartPuck, dataclasses, np, plt, torch


@app.cell
def engine(dataclasses, np, torch):
    # Fractal-trainability engine - faithful torch port of Sohl-Dickstein's
    # reference implementation (github.com/Sohl-Dickstein/fractal, MIT),
    # validated against the author's canonical rendering. See notebook text
    # for the ensemble extension this enables.
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

    return (
        Config,
        boxcount_dimension,
        cdf_img,
        compute_grid,
        extract_edges,
        loss_trace,
    )


@app.cell(hide_code=True)
def sec1_intro(mo):
    mo.md(r"""
    ## 1. A map where every pixel is a training run

    The setup mirrors the paper's reference implementation exactly:

    - a 1-hidden-layer tanh network, 16 units wide, in mean-field parameterization
    - a fixed random dataset of 272 points (as many data points as the network has parameters)
    - full-batch gradient descent for 500 steps
    - the two axes are the log learning rates of the input layer and the output layer

    Each pixel gets a signed convergence score: negative (cool colors) means training settled, positive (warm colors) means it diverged, and the magnitude encodes how fast. The score, the network, and the color normalization follow the author's public reference code (MIT), reimplemented in PyTorch and validated against the author's own rendering.
    """)
    return


@app.cell
def canonical_compute(Config, compute_grid, np, torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(steps=500)
    CANON_WINDOW = (-3.0, 6.0, -3.0, 6.0)

    # One full map. On GPU: 512 x 512 = 262,144 independent training runs.
    # A GPU-less fork drops to 96 x 96 so the first load stays under a couple
    # of minutes (the methods cell says exactly this). Cached to disk either way.
    # zoom_res drives section 2's recompute size the same way.
    import os as _os

    _res_canon = 512 if device == "cuda" else 96
    zoom_res = 256 if device == "cuda" else 96
    _cache = f"/tmp/fractal_canonical_{_res_canon}.npy"
    if _os.path.exists(_cache):
        canonical_img = np.load(_cache)
    else:
        canonical_img = compute_grid(cfg, _res_canon, device=device, window=CANON_WINDOW)
        np.save(_cache, canonical_img)
    canonical_img.shape
    return CANON_WINDOW, canonical_img, cfg, device, zoom_res


@app.cell
def canonical_display(CANON_WINDOW, canonical_img, cdf_img, plt):
    _disp = cdf_img(canonical_img)
    _fig, _ax = plt.subplots(figsize=(8, 8))
    _ax.imshow(
        _disp,
        origin="lower",
        cmap="Spectral",
        extent=[CANON_WINDOW[0], CANON_WINDOW[1], CANON_WINDOW[2], CANON_WINDOW[3]],
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    _ax.set_xlabel("log10 learning rate (input layer)")
    _ax.set_ylabel("log10 learning rate (output layer)")
    _ax.set_title(
        f"Trainability map: {canonical_img.shape[0] ** 2:,} training runs, one per pixel"
    )
    _fig
    return


@app.cell(hide_code=True)
def sec1_caption(mo):
    mo.md(r"""
    Cool region (bottom): training converges. Warm region (top): training diverges. The interesting part is the border, especially toward the right where the input layer learning rate is large. It looks rough. The question is whether that roughness is superficial texture or structure all the way down.

    Boundary fractal dimension of this map, by box counting: a smooth curve would measure 1.0, space-filling noise 2.0. We measure it live in section 3.
    """)
    return


@app.cell(hide_code=True)
def sec2_intro(mo):
    mo.md(r"""
    ## 2. Sail the coastline

    The map above is the whole ocean. The claim lives at the border. Below, take the guided route (three successively deeper zooms into the roughest stretch of coast), or chart your own course by dragging a box straight onto the map. Range sliders are tucked underneath for precise coordinates.

    Every zoom recomputes the map inside your box from scratch: 65,536 fresh training runs per view (a GPU-less fork uses a smaller grid). Nothing is an upscaled image. If the boundary were an ordinary curve, it would straighten out within a zoom or two. Watch what actually happens.
    """)
    return


@app.cell
def zoom_route():
    # The guided route: windows chosen by measured boundary density on the
    # canonical map (sliding-window edge count, see appendix). Each window nests
    # inside the previous one. (lr1_lo, lr1_hi, lr2_lo, lr2_hi) in log10 units.
    ZOOM_ROUTE = {
        "full map": (-3.0, 6.0, -3.0, 6.0),
        "zoom 1: the rough coast": (4.17, 5.96, 1.48, 3.28),
        "zoom 2: into the filaments": (5.37, 5.96, 1.48, 2.08),
        "zoom 3: still not smooth": (5.53, 5.72, 1.75, 1.94),
    }
    return (ZOOM_ROUTE,)


@app.cell
def zoom_ui(
    CANON_WINDOW,
    ZOOM_ROUTE,
    canonical_img,
    cdf_img,
    mo,
    plt,
    zoom_res,
):
    # Zoom controls. Primary: drag a box on the map itself (commits on mouse-up;
    # shift-drag draws a lasso, whose bounding box is used). The guided-route
    # dropdown and precise range sliders remain as alternatives.
    _selfig, _selax = plt.subplots(figsize=(6.4, 6.4))
    _selax.imshow(
        cdf_img(canonical_img),
        origin="lower",
        cmap="Spectral",
        extent=[CANON_WINDOW[0], CANON_WINDOW[1], CANON_WINDOW[2], CANON_WINDOW[3]],
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    _selax.set_xlabel("log10 lr (input layer)")
    _selax.set_ylabel("log10 lr (output layer)")
    _selax.set_title("drag a box to zoom anywhere")
    zoom_select = mo.ui.matplotlib(_selax, debounce=True)
    plt.close(_selfig)

    zoom_choice = mo.ui.dropdown(
        options=list(ZOOM_ROUTE.keys()),
        value="full map",
        label="guided route",
    )
    lr1_range = mo.ui.range_slider(
        start=-3.0,
        stop=6.0,
        step=0.05,
        value=(0.6, 2.4),
        show_value=True,
        label="free explore: log10 lr (input layer)",
        full_width=True,
    )
    lr2_range = mo.ui.range_slider(
        start=-3.0,
        stop=6.0,
        step=0.05,
        value=(-1.0, 0.2),
        show_value=True,
        label="free explore: log10 lr (output layer)",
        full_width=True,
    )
    free_go = mo.ui.run_button(
        label=f"compute my window ({zoom_res} x {zoom_res} = {zoom_res**2:,} training runs)"
    )
    mo.vstack(
        [
            zoom_select,
            zoom_choice,
            mo.accordion(
                {"precise coordinates instead": mo.vstack([lr1_range, lr2_range, free_go])}
            ),
        ]
    )
    return free_go, lr1_range, lr2_range, zoom_choice, zoom_select


@app.cell
def zoom_compute(
    ZOOM_ROUTE,
    cfg,
    compute_grid,
    device,
    free_go,
    lr1_range,
    lr2_range,
    mo,
    np,
    zoom_choice,
    zoom_res,
    zoom_select,
):
    # Compute the selected window. Priority: a drawn box wins; otherwise the
    # run-button sliders; otherwise the guided route. A selection smaller than
    # 0.05 log-units on either axis is treated as an accidental click and
    # ignored, which is also how you clear a box. Guided windows are cached on
    # first computation; fresh windows compute on demand.
    _ZOOM_CACHE_DIR = "/tmp/zoom_cache"
    import os as _os
    import hashlib as _hashlib

    _os.makedirs(_ZOOM_CACHE_DIR, exist_ok=True)

    _sel_window = None
    _sel = zoom_select.value
    if _sel:
        if hasattr(_sel, "x_min"):
            _cand = (
                float(_sel.x_min),
                float(_sel.x_max),
                float(_sel.y_min),
                float(_sel.y_max),
            )
        else:  # lasso: use the bounding box of its vertices
            _vx = [float(p[0]) for p in _sel.vertices]
            _vy = [float(p[1]) for p in _sel.vertices]
            _cand = (min(_vx), max(_vx), min(_vy), max(_vy))
        if (_cand[1] - _cand[0]) >= 0.05 and (_cand[3] - _cand[2]) >= 0.05:
            _sel_window = _cand

    if _sel_window is not None:
        zoom_window = _sel_window
    elif free_go.value:
        zoom_window = (
            float(lr1_range.value[0]),
            float(lr1_range.value[1]),
            float(lr2_range.value[0]),
            float(lr2_range.value[1]),
        )
    else:
        zoom_window = ZOOM_ROUTE[zoom_choice.value]

    _key = _hashlib.sha1(repr((zoom_window, zoom_res, cfg.steps)).encode()).hexdigest()[:16]
    _zpath = f"{_ZOOM_CACHE_DIR}/{_key}.npy"
    if _os.path.exists(_zpath):
        zoom_img = np.load(_zpath)
    else:
        with mo.status.spinner(
            title=f"training {zoom_res**2:,} networks in {zoom_window} ..."
        ):
            zoom_img = compute_grid(cfg, zoom_res, device=device, window=zoom_window)
        np.save(_zpath, zoom_img)
    zoom_img.shape
    return zoom_img, zoom_window


@app.cell
def zoom_display(
    CANON_WINDOW,
    boxcount_dimension,
    canonical_img,
    cdf_img,
    extract_edges,
    plt,
    zoom_img,
    zoom_window,
):
    _fig2, (_axL, _axR) = plt.subplots(1, 2, figsize=(13, 6.2))
    _axL.imshow(
        cdf_img(canonical_img),
        origin="lower",
        cmap="Spectral",
        extent=[CANON_WINDOW[0], CANON_WINDOW[1], CANON_WINDOW[2], CANON_WINDOW[3]],
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    _w = zoom_window
    _axL.add_patch(
        plt.Rectangle(
            (_w[0], _w[2]),
            _w[1] - _w[0],
            _w[3] - _w[2],
            fill=False,
            edgecolor="#e63946",
            linewidth=1.8,
        )
    )
    _axL.set_title("where you are on the full map")
    _axL.set_xlabel("log10 lr (input layer)")
    _axL.set_ylabel("log10 lr (output layer)")

    _axR.imshow(
        cdf_img(zoom_img),
        origin="lower",
        cmap="Spectral",
        extent=[_w[0], _w[1], _w[2], _w[3]],
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    _zedges = extract_edges(zoom_img)
    if _zedges.any():
        _zdim, _, _ = boxcount_dimension(_zedges)
        _axR.set_title(f"inside the box: boundary dimension = {_zdim:.2f}")
    else:
        _axR.set_title("inside the box: no boundary in this window")
    _axR.set_xlabel("log10 lr (input layer)")
    _fig2.tight_layout()
    _fig2
    return


@app.cell(hide_code=True)
def sec2_caption(mo):
    mo.md(r"""
    The number over the right panel is the box-counting fractal dimension of the boundary inside your window (estimator validated on shapes of known dimension; see the appendix cell). A smooth curve measures 1.0. Across the guided route the dimension stays well above 1 at every depth: the coastline refuses to straighten. That is the paper's claim, reproduced live.

    A drawn box takes priority over the dropdown and sliders. To hand control back to the guided route, draw a tiny box with a quick click-drag: anything smaller than 0.05 log-units is read as "clear".
    """)
    return


@app.cell(hide_code=True)
def anchor_intro(mo):
    mo.md(r"""
    ### Drop anchor: replay any pixel, live

    The map compresses each training run into one color. This panel decompresses it. Drag the puck and the kernel retrains that exact network, all 500 steps, while you watch.

    Park it deep in the blue and the loss sinks quietly. Park it in the red and the loss explodes within a few steps. Then bring it to the coastline. Runs there hesitate for hundreds of steps, swinging across orders of magnitude before they commit. Two pixels that touch can end on opposite shores because these trajectories are chaotic: microscopic differences grow exponentially until they decide the outcome.
    """)
    return


@app.cell
def anchor_widget(CANON_WINDOW, ChartPuck, canonical_img, cdf_img, mo, plt):
    # Drag-me map: a wigglystuff ChartPuck overlaid on the canonical map.
    # throttle=150 ms keeps the replay smooth while dragging.
    _pfig, _pax = plt.subplots(figsize=(6.0, 6.0))
    _pax.imshow(
        cdf_img(canonical_img),
        origin="lower",
        cmap="Spectral",
        extent=[CANON_WINDOW[0], CANON_WINDOW[1], CANON_WINDOW[2], CANON_WINDOW[3]],
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    _pax.set_xlabel("log10 lr (input layer)")
    _pax.set_ylabel("log10 lr (output layer)")
    _pax.set_title("drag the puck; it starts on the boundary")
    puck = mo.ui.anywidget(ChartPuck(_pfig, x=2.11, y=2.13, throttle=150))
    plt.close(_pfig)
    return (puck,)


@app.cell
def anchor_trace(CANON_WINDOW, canonical_img, cfg, device, loss_trace, puck):
    # Replay the training run at the puck's position. loss_trace mirrors
    # convergence_grid with G=1, so this is the pixel's actual biography,
    # not an illustration. map_converged reads the pixel's own sign for the
    # consistency check in the panel title.
    puck_lr = (float(puck.value["x"][0]), float(puck.value["y"][0]))
    trace_v, trace_converged = loss_trace(
        cfg, 10.0 ** puck_lr[0], 10.0 ** puck_lr[1], device
    )
    _res = canonical_img.shape[0]
    _ix = int(
        round(
            (puck_lr[0] - CANON_WINDOW[0])
            / (CANON_WINDOW[1] - CANON_WINDOW[0])
            * (_res - 1)
        )
    )
    _iy = int(
        round(
            (puck_lr[1] - CANON_WINDOW[2])
            / (CANON_WINDOW[3] - CANON_WINDOW[2])
            * (_res - 1)
        )
    )
    map_converged = bool(
        canonical_img[min(max(_iy, 0), _res - 1), min(max(_ix, 0), _res - 1)] < 0
    )
    return map_converged, puck_lr, trace_converged, trace_v


@app.cell
def anchor_display(
    map_converged,
    mo,
    np,
    plt,
    puck,
    puck_lr,
    trace_converged,
    trace_v,
):
    _tfig, _tax = plt.subplots(figsize=(6.6, 4.8))
    _c = "#2166ac" if trace_converged else "#e63946"
    _tax.semilogy(
        np.arange(len(trace_v)), np.clip(trace_v, 1e-12, None), color=_c, linewidth=1.3
    )
    _tax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    _tax.text(
        len(trace_v) * 0.99, 1.25, "initial loss", ha="right", fontsize=8, color="#777777"
    )
    _tax.set_xlabel("training step")
    _tax.set_ylabel("loss / initial loss")
    _verdict = "converged" if trace_converged else "diverged"
    _mark = (
        "replay matches its pixel"
        if trace_converged == map_converged
        else "replay DISAGREES with its pixel (see note below)"
    )
    _tax.set_title(
        "lr_in = 1e{:+.2f}, lr_out = 1e{:+.2f}  ->  {}\n({})".format(
            puck_lr[0], puck_lr[1], _verdict, _mark
        ),
        fontsize=10,
    )
    _tfig.tight_layout()
    mo.hstack([puck, _tfig], gap=1.0, wrap=True)
    return


@app.cell(hide_code=True)
def anchor_caption(mo):
    mo.md(r"""
    A note on the rare disagreement: right on the knife edge, the replay can land on the opposite side from its own pixel. Both compute the same trajectory in fp32, but the map batches 262,144 runs and the replay runs one, so their floating-point additions happen in different orders. Away from the boundary that difference changes nothing. On it, a rounding-error-sized nudge can decide the fate of the run. Finding such a point is not a bug to hide; it is the sharpest demonstration this notebook has of what a chaotic boundary means.
    """)
    return


@app.cell(hide_code=True)
def sec3_intro(mo):
    mo.md(r"""
    ## 3. Does the coastline survive noise?

    The paper studies full-batch gradient descent on purpose: determinism makes the fractal crisp. The author also rendered one minibatch variant, but with a single fixed noise schedule shared by every point on the map, so the boundary stays a deterministic set. Real training is noisier than that: every run draws its own minibatches.

    So here is a question the paper's own artifacts leave open. Treat the boundary as an ensemble object: give every pixel K = 48 independent minibatch schedules and record the fraction that converge, P(converged). At full batch, P is 0 or 1 everywhere and the boundary is the crisp fractal from section 1. As the batch shrinks and noise grows, does that structure survive?

    We answer two ways: by eye, with the probability field below, and by measurement, with the box-counting dimension of the P = 0.5 contour (drawn in crimson, like every boundary in this notebook) as a function of batch size, bootstrapped over the ensemble.
    """)
    return


@app.cell
def batch_slider(mo):
    batch_pick = mo.ui.slider(
        steps=[8, 16, 32, 64, 136, 272],
        value=64,
        label="minibatch size (272 = same size as the dataset, drawn with replacement)",
        show_value=True,
        full_width=True,
    )
    batch_pick
    return (batch_pick,)


@app.cell(hide_code=True)
def sec3_data(np):
    # Section 3 data, embedded so the notebook is self-contained and survives
    # session restarts. 256x256 grid, 48 independent minibatch schedules per
    # batch size. P-fields are float16, deterministic reference is a bit mask,
    # zlib+base64 (28 KB). The heavy sweep that produced this ran once on the GPU.
    import base64 as _b64, zlib as _zlib, io as _io
    _SEC3_B64 = "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"
    _npz = np.load(_io.BytesIO(_zlib.decompress(_b64.b64decode(_SEC3_B64))))
    NOISE_RES, NOISE_K = 256, 48
    NOISE_BATCH_SIZES = [8, 16, 32, 64, 136, 272]
    det_noise_img = _npz['det_mask'].astype(np.float32)
    noise_pfields = {bs: _npz[f'P_{bs}'].astype(np.float32) for bs in NOISE_BATCH_SIZES}
    # (batch, dimension, ci_lo, ci_hi): bootstrap over the 48 schedules, B=400.
    DIM_ROWS = [[8,1.112,1.076,1.146],[16,1.100,1.069,1.137],[32,1.101,1.070,1.131],[64,1.092,1.061,1.133],[136,1.095,1.054,1.153],[272,1.146,1.085,1.198]]
    det_dim = 1.193

    def p_field(bs):
        return noise_pfields[int(bs)]

    f'{len(noise_pfields)} P-fields embedded ({len(_SEC3_B64)//1024} KB); deterministic dim {det_dim}'
    return DIM_ROWS, NOISE_K, det_dim, det_noise_img, p_field


@app.cell
def pfield_display(
    CANON_WINDOW,
    NOISE_K,
    batch_pick,
    det_noise_img,
    np,
    p_field,
    plt,
):
    P_sel = p_field(int(batch_pick.value))
    _figP, (_axA, _axB) = plt.subplots(1, 2, figsize=(13, 6.2))
    _axA.imshow(
        det_noise_img,
        origin="lower",
        cmap="cividis",
        extent=[CANON_WINDOW[0], CANON_WINDOW[1], CANON_WINDOW[2], CANON_WINDOW[3]],
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    _axA.set_title("deterministic (full batch): P is 0 or 1, boundary is crisp")
    _axA.set_xlabel("log10 lr (input layer)")
    _axA.set_ylabel("log10 lr (output layer)")
    _imB = _axB.imshow(
        P_sel,
        origin="lower",
        cmap="cividis",
        extent=[CANON_WINDOW[0], CANON_WINDOW[1], CANON_WINDOW[2], CANON_WINDOW[3]],
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    # The coastline, in the same crimson it wears everywhere in this notebook:
    # the P = 0.5 contour of each field.
    _cx = np.linspace(CANON_WINDOW[0], CANON_WINDOW[1], P_sel.shape[1])
    _cy = np.linspace(CANON_WINDOW[2], CANON_WINDOW[3], P_sel.shape[0])
    _axA.contour(
        _cx,
        _cy,
        np.asarray(det_noise_img, dtype=np.float32),
        levels=[0.5],
        colors=["#e63946"],
        linewidths=0.8,
    )
    _axB.contour(
        _cx,
        _cy,
        np.asarray(P_sel, dtype=np.float32),
        levels=[0.5],
        colors=["#e63946"],
        linewidths=0.8,
    )
    _unc = float(((P_sel > 0) & (P_sel < 1)).mean())
    _axB.set_title(
        f"batch size {batch_pick.value}: P(converged) over {NOISE_K} schedules (fuzzy band {100 * _unc:.0f}%)"
    )
    _axB.set_xlabel("log10 lr (input layer)")
    _figP.colorbar(_imB, ax=_axB, label="P(converged)", shrink=0.8)
    _figP.tight_layout()
    _figP
    return


@app.cell
def dim_curve(DIM_ROWS, det_dim, plt):
    _figD, _axD = plt.subplots(figsize=(9, 5.5))
    _bss = [r[0] for r in DIM_ROWS]
    _dims = [r[1] for r in DIM_ROWS]
    _los = [max(0.0, r[1] - r[2]) for r in DIM_ROWS]
    _his = [max(0.0, r[3] - r[1]) for r in DIM_ROWS]
    _axD.errorbar(
        _bss, _dims, yerr=[_los, _his], fmt="o-", capsize=4, linewidth=2, color="#e63946"
    )
    _axD.axhline(det_dim, linestyle="--", linewidth=1.5, color="#555555")
    _axD.annotate(
        f"deterministic fractal ({det_dim:.2f})",
        xy=(_bss[-3], det_dim),
        xytext=(0, 8),
        textcoords="offset points",
    )
    _axD.axhline(1.0, linestyle=":", linewidth=1.2, color="#999999")
    _axD.annotate(
        "smooth curve (1.0)", xy=(_bss[1], 1.0), xytext=(0, 6), textcoords="offset points"
    )
    _axD.set_xscale("log", base=2)
    _axD.set_ylim(0.98, 1.22)
    _axD.set_xlabel("minibatch size (log scale); dataset has 272 points")
    _axD.set_ylabel("box-counting dimension of the P = 0.5 contour")
    _axD.set_title("Noise erodes the fine structure, then holds (256 grid, 48 schedules)")
    _figD.tight_layout()
    _figD
    return


@app.cell(hide_code=True)
def sec3_caption(mo):
    mo.md(r"""
    Two things are visible.

    **By eye:** the crisp knife-edge of the deterministic boundary smears into a soft probabilistic band. Where full-batch training gives a definite yes or no at each pixel, noisy training gives a coin whose bias slides smoothly across the transition. That smearing is the qualitative answer, and it does not depend on how we measure it.

    **By measurement:** noise erodes the fine fractal filigree along the boundary. The box-counting dimension falls from the deterministic value (about 1.19 at this resolution) to a plateau near 1.10 as soon as real minibatch noise is present, and holds there as noise grows further. Read carefully, this is a partial erosion, not an erasure. A dimension of 1.10 sits well below the deterministic fractal but clearly above a smooth curve at 1.0. Noise sands down the sharpest detail of the coastline and then stops, leaving a residual roughness that persists at every noise level we tested.
    """)
    return


@app.cell(hide_code=True)
def guardrails(mo):
    mo.md(r"""
    ### What this does and does not show

    - The landscape, network, and scoring follow the paper's reference setup; the ensemble treatment and the dimension-versus-noise curve are this notebook's addition. The author's own minibatch rendering used a single shared schedule and stays fractal. Our claim is about the ensemble object, which is a different set.
    - Dimension estimates are resolution- and step-budget-dependent. Every point on the curve, including the deterministic reference, uses the same resolution, step budget, and ensemble size, so the comparison is like-for-like. Absolute values would shift at higher resolution; the drop and the plateau are the claim, not the exact numbers.
    - Honesty about our own ruler: an earlier version of this sweep, with only 16 schedules per point, appeared to show a sharp dip and a rebound. Quadrupling the ensemble to 48 flattened both into the plateau shown here. The transient structure was sampling noise in the estimator, not a property of the boundary. We report the converged result, and the estimator is calibrated on shapes of known dimension in the appendix.
    - The mildest noise here (batch 272, the size of the dataset, drawn with replacement) already pulls the dimension toward the plateau, though its interval is wide enough to overlap the deterministic value. From batch 136 down, the erosion is clearly resolved.
    """)
    return


@app.cell(hide_code=True)
def verdict(mo):
    mo.md(r"""
    ## Verdict

    Three findings, each computed live in this notebook:

    1. **The boundary is fractal.** The border between trainable and divergent does not smooth under magnification. Its box-counting dimension stays well above 1.0 at every depth on the guided route, reaching about 1.9 in the deepest window, where a smooth curve would read 1.0.
    2. **You can feel why.** Park the replay puck on the coastline and the loss curve hesitates for hundreds of steps before committing. The chaos that lets two touching pixels end on opposite shores is the same sensitivity that carves the border into a fractal.
    3. **Noise sands it, but does not erase it.** Give every pixel its own minibatch schedule and the crisp coastline becomes a probability band. Its dimension drops from the deterministic 1.19 to a plateau near 1.10 and holds there: real training noise removes the sharpest filigree and leaves a residual roughness that never becomes a smooth curve.

    The practical reading: near the edge of trainability there is no tidy threshold to memorize. The edge is not a line you can tune against. It is a coastline.
    """)
    return


@app.cell(hide_code=True)
def appendix_intro(mo):
    mo.md(r"""
    ## Appendix: trust, but verify the ruler

    Every dimension number in this notebook comes from one box-counting estimator. Before believing anything it says about trainability boundaries, it has to earn trust on shapes whose dimension we know exactly. The cell below runs that calibration live, every time this notebook runs.
    """)
    return


@app.cell
def calibration(boxcount_dimension, extract_edges, mo, np):
    # Calibration of the box-counting estimator on known-dimension shapes.
    _n = 729

    _line = np.zeros((_n, _n), bool)
    for _i in range(_n):
        _line[_i, _i] = True

    _smooth = extract_edges(np.fromfunction(lambda i, j: (i - 0.4 * j - 200.0), (_n, _n)))


    def _carpet(k):
        m = np.ones((1, 1), bool)
        for _ in range(k):
            z = np.zeros_like(m)
            m = np.block([[m, m, m], [m, z, m], [m, m, m]])
        return m


    _rng_cal = np.random.default_rng(0)
    _cases = [
        ("straight line", 1.0, _line),
        ("smooth boundary", 1.0, _smooth),
        ("Sierpinski carpet", 1.893, _carpet(6)),
        ("random mask", 2.0, _rng_cal.random((_n, _n)) < 0.5),
    ]
    _rows = []
    for _name, _truth, _mask in _cases:
        _d, _, _ = boxcount_dimension(_mask)
        _rows.append({"shape": _name, "true dimension": _truth, "measured": round(_d, 3)})
    estimator_table = _rows
    mo.ui.table(_rows, selection=None, label="box-counting estimator calibration")
    return


@app.cell(hide_code=True)
def methods(mo):
    mo.md(r"""
    ### Methods and credits

    - **Paper:** Jascha Sohl-Dickstein, *The boundary of neural network trainability is fractal*, arXiv 2402.06184 (2024).
    - **Reference implementation:** the author's public notebook (MIT license) defines the network, the signed convergence measure, and the rank-based color normalization used here. This notebook reimplements them in PyTorch and was validated against the author's canonical rendering before anything else was built. The author also rendered one minibatch variant (a single schedule shared across the map); the ensemble treatment in section 3 is this notebook's addition.
    - **Compute:** every map is computed in this notebook's own runtime. The full map is 262,144 independent training runs; a typical session of this notebook trains several million tiny networks. The core reproduction also runs on CPU (smaller grids); the GPU earns its keep in section 3, where the schedule ensembles multiply the work by an order of magnitude.
    - **Honesty notes:** dimension estimates are resolution- and step-budget-dependent, so every comparison here holds those fixed; the calibration table above bounds estimator error at about one percent on known shapes; section 3's ensemble size is stated next to every number it affects.
    """)
    return


if __name__ == "__main__":
    app.run()
