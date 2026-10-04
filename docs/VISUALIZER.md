# The 3D model visualizer

An interactive view of a trained (or training) GW1B model in the spirit of [bbycroft.net/llm](https://bbycroft.net/llm):
the architecture as a tower of slabs — token embedding at the bottom, one group per transformer block (norm · Q K V ·
O · norm · gate up · down), final norm and output head on top — with every slab coloured by the **actual weights of a
checkpoint**. Rotate, pan and zoom; hover a slab for its shape, statistics, histogram and a magnified view; click to fly
to it. Pick the checkpoint step, or follow the latest one while the run trains.

![what you get](img/visualizer.png)

## Open it (while a run trains, or afterwards)

```bash
gw1b jupyter          # if you do not have a session yet: the tunnel it prints forwards port 6007 as well
gw1b viz              # on the login node; then open  http://localhost:6007  on your laptop
```

`gw1b viz` starts a small server inside your Jupyter job (CPU only — it never touches the GPU) over
`$GW1B_SCRATCH/users/<netid>/runs`. The page lists the runs that have checkpoints; the first export of a checkpoint
takes a few seconds for the proxy models and about a minute for GW-1B, after that it is cached
(`<run>/viz/step-<N>-vs-<M>.json`). With **follow latest** ticked the page checks every minute and reloads when the
training job has written a new checkpoint (`run.ckpt_every` steps), keeping your camera where it was.

Other directories: `gw1b viz /scratch/gw1b-class/teams/team3 $GW1B_SCRATCH/users/<netid>/runs`.

## What the colours mean

Every parameter matrix is reduced to at most 48 × 48 cells (each cell is a block of weights):

| show | cell value | colour map | use it for |
|---|---|---|---|
| **weight magnitude** | RMS of the block | plasma (dark → bright) | where the large weights live; the structure of embeddings and projections |
| **weight sign** | mean of the block | blue – white – red | systematic positive/negative regions (e.g. in norm scales, biases of structure) |
| **change since previous checkpoint** | RMS of (w<sub>now</sub> − w<sub>prev</sub>) per block | heat | which layers are still learning; what a learning-rate decay or a warm restart does |

**scale** = *per tensor* stretches each slab from its own min to max (shows structure inside a matrix); *global* uses
one scale for all matrices (compares layers; the embedding then dominates, as it should). Norm scales (vectors) are the
thin strips to the left of the slabs they precede. The info panel also gives the relative change
`rms(Δ) / rms(w)` — a quick "how much did this tensor move" number per layer.

## A page you can send around

```bash
gw1b python -m gw1b.viz.export --run $GW1B_SCRATCH/users/$USER/runs/50m --html 50m-step4000.html
gw1b python -m gw1b.viz.export --run ... --step 2000 --compare 1000 --html diff.html   # a specific pair of steps
```

`--html` writes one self-contained file (~1 MB + the data) that works from a laptop without any server or internet:
three.js and the data are inlined. `--out` writes just the JSON (`<run>/viz/step-<N>.json` by default).

## How it works

* `gw1b/viz/export.py` reads one checkpoint step with Orbax (as plain arrays, no model object), computes per-tensor
  statistics and histograms, the block-RMS / block-mean tiles and, against the previous step, the block RMS of the change.
* `gw1b/viz/server.py` is a stdlib HTTP server: `/api/runs` (runs + steps + last metrics), `/api/export?run=&step=&compare=`
  (exports on demand, caches on disk), and the static page.
* `gw1b/viz/static/` is the page: `index.html`, `viz.js` (three.js r160 + OrbitControls, vendored under `vendor/`, MIT).
  Layout: slab width/depth = `0.75·log2(dim)`, so a 32000 × 512 embedding and a 512 × 512 projection fit in one view.

Ideas students have asked for and can add in `viz.js`/`export.py`: per-head slicing of Q/K/V, attention-head similarity,
the optimizer moments (`opt` in the checkpoint), activations for a prompt (`model.hidden`), and a time slider over
all checkpoints (export each step once; the page already keeps the camera).
