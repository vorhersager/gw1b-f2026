# Using Google Colab as the front end for Pegasus GPUs

Colab can drive **any** Jupyter server as a "local runtime" — the UI runs at colab.research.google.com,
the kernel runs on the Pegasus GPU node. Students get the Colab interface they know (and Drive-hosted
notebooks) while every computation, file and checkpoint stays on Pegasus.

## Steps

1. On Pegasus: `gw1b jupyter` (or from the laptop: `laptop/gw1b-connect.sh <netid>`). Wait for the connection lines.
2. On your laptop: run the printed tunnel command and keep that terminal open:
   `ssh -N -L 8888:<node>:<port> -L 6006:<node>:6006 <netid>@pegasus.arc.gwu.edu`
3. In Colab: **Connect ▾ → Connect to a local runtime**, paste `http://localhost:8888/?token=<token>` (printed by
   `gw1b jupyter`), click **Connect**. The status bar shows the connection; `!hostname` prints the GPU node name.

That is it — `!nvidia-smi`, `import jax; jax.devices()`, all the `gw1b` notebooks work unchanged.

## What is different from a normal Colab

| | Google-hosted Colab runtime | Colab connected to Pegasus |
|---|---|---|
| where code runs | Google VM (T4/A100 if paid) | Pegasus GPU node from your Slurm job |
| files | `/content`, lost when the VM dies | your Pegasus home + `$GW1B_SCRATCH`, persistent |
| `pip install` | works | disabled on purpose (shared, pinned environment) — ask the instructor |
| Drive mount (`drive.mount`) | works | not available (the runtime is not a Google VM); use Pegasus storage, `scp`, or GitHub |
| session limit | ~12 h / idle disconnects | your job's `--time` (default 4 h); the tunnel must stay open |
| internet from the kernel | yes | Pegasus compute nodes may have no outbound internet → datasets are pre-downloaded to `$HF_HOME` |

Notebooks saved "in Drive" are just the `.ipynb` file on Google's side; the data stays on Pegasus. If you
prefer to keep notebooks on Pegasus too, use the JupyterLab URL instead of Colab — same kernel, same job.

## How it works / security

`slurm/jupyter.sbatch` starts JupyterLab with
`--ServerApp.allow_origin='https://colab.research.google.com' --ServerApp.allow_credentials=True`
(what Colab's [local-runtime instructions](https://research.google.com/colaboratory/local-runtimes.html)
require; the old `jupyter_http_over_ws` extension is no longer needed), bound to the compute node's
interface on a random port with a random 48-hex-character token. Only your ssh tunnel reaches it
(the node is not exposed outside the cluster), and the token is required for every request.

## Troubleshooting

* *"Unable to connect to the runtime"* — is the tunnel terminal still open? Did you paste the URL **with** the token?
  Try the JupyterLab URL in a normal browser tab first; if that works, Colab will too.
* *Connected, but the notebook cannot see `gw1b`* — the kernel started outside the environment. In Colab pick
  **Runtime → Change runtime type** and make sure you are connected to the local runtime (not a hosted one);
  the first notebook cell (`00_hello_pegasus`) prints the JAX devices as a check.
* *Port 8888 already in use on the laptop* — `gw1b jupyter --local-port 8890` (the printed lines change accordingly).
* *2FA prompts twice* — normal with Windows OpenSSH (one login to start the job, one for the tunnel);
  on macOS/Linux `gw1b-connect.sh` reuses the first connection.
