# The Coastline That Never Smooths

A marimo notebook built on "The boundary of neural network trainability is
fractal" (Jascha Sohl-Dickstein, 2024, arXiv 2402.06184), for the
marimo x alphaXiv Notebook Competition #2.

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
