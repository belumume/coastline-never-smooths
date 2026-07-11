# The Coastline That Never Smooths

A marimo notebook built on "The boundary of neural network trainability is
fractal" (Jascha Sohl-Dickstein, 2024, arXiv 2402.06184), for the
alphaXiv x marimo molab Notebook Competition #2.

Every pixel of the map is a real gradient descent training run. Drag a box to
zoom anywhere (65,536 fresh runs per view), drop the puck on the boundary to
replay a single pixel's training live, then watch what minibatch noise does to
the coastline's fractal dimension.

Live notebook: https://molab.marimo.io/notebooks/nb_SZ6ifRSGVy1kQY8b5yXrax

Run locally:

    uvx marimo edit --sandbox notebook.py

A GPU helps (the full map is 262,144 training runs); a CPU-only machine drops
to a smaller grid on its own.

`engine.py` is the standalone version of the notebook's core, validated
against the author's reference implementation (MIT). `__marimo__/` carries
rendered outputs so the notebook shows results even before a re-run.

## What this adds to the paper

The paper's minibatch experiment shares one fixed noise schedule across the whole
map (`PRNGKey(42)` split per step, shared across the vmapped grid), so its noisy
boundary is still a deterministic set, which is why it stays fractal.

This notebook gives every training run its own independent noise draw. The
boundary becomes a probability object, and its 50% survival contour is measured
instead: the box-counting dimension falls from about 1.19 to a plateau near 1.10
and holds there as noise grows. Partial erosion, not smoothing. Noise sands off the finest filigree, and a
residual roughness survives every level tested.

Honesty notes are in the notebook, not buried: dimension estimates are
resolution-dependent (every comparison here holds resolution, step budget, and
ensemble size fixed), the estimator is calibrated on shapes of known dimension
first (line 1.009, circle 1.005, Sierpinski carpet 1.903), and the mildest noise
level is *not* claimed because its interval still overlaps the deterministic value.

## The trilogy

Three notebooks, one per paper, three views of the same edge:

- **The Coastline That Never Smooths** (this one): the stability boundary in hyperparameter space is fractal.
- **On the Wire**: training *rides* that boundary. https://molab.marimo.io/notebooks/nb_QtxPsWxaZSpnRU2J8qWZWm
- **The Scale DyT Forgot**: a frozen scalar pins where that boundary sits relative to your input scale. https://molab.marimo.io/notebooks/nb_jNm948dpXRCvrK9demdeaf
