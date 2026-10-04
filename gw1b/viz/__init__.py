"""Interactive 3D visualizer of a trained GW1B model (architecture + weights from checkpoints).

    python -m gw1b.viz.export --run $GW1B_SCRATCH/users/$USER/runs/50m            # -> <run>/viz/step-<N>.json
    python -m gw1b.viz.export --run ... --html model.html                          # self-contained page (share it)
    python -m gw1b.viz.server --runs $GW1B_SCRATCH/users/$USER/runs --port 6007    # live page, refreshes as the run trains
    gw1b viz                                                                       # the server inside your Jupyter job

The exporter reduces every weight matrix to a small tile (block RMS / block mean, and the change since the previous
checkpoint) plus statistics; the page (gw1b/viz/static) draws the model as stacked 3D slabs with three.js.
"""
