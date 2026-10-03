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

## The class notebooks, one click away

Each badge opens the notebook from GitHub in Colab (the repository is public). Then do step 3 above — the
badge alone gives you a Google-hosted runtime without the class environment; the first cell says so if you
forget. Colab saves your copy to Drive; the data and checkpoints stay on Pegasus.

| notebook | what it covers | |
|---|---|---|
| `00_hello_pegasus` | GPU check, where files live, your compute budget | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorhersager/gw1b-f2026/blob/main/notebooks/00_hello_pegasus.ipynb) |
| `01_tokenizer` | train BPE vs unigram tokenizers, compare compression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorhersager/gw1b-f2026/blob/main/notebooks/01_tokenizer.ipynb) |
| `02_data_pipeline` | documents → token shards → batches | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorhersager/gw1b-f2026/blob/main/notebooks/02_data_pipeline.ipynb) |
| `03_train_proxy_model` | train a small model end-to-end, read `metrics.jsonl`, resume | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorhersager/gw1b-f2026/blob/main/notebooks/03_train_proxy_model.ipynb) |
| `04_evaluate_and_export` | perplexity, lm-eval benchmarks, export to Hugging Face format | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorhersager/gw1b-f2026/blob/main/notebooks/04_evaluate_and_export.ipynb) |


## What is different from a normal Colab

| | Google-hosted Colab runtime | Colab connected to Pegasus |
|---|---|---|
| where code runs | Google VM (T4/A100 if paid) | Pegasus GPU node from your Slurm job |
| files | `/content`, lost when the VM dies | your Pegasus home + `$GW1B_SCRATCH`, persistent |
| `pip install` | works | disabled on purpose (shared, pinned environment) — ask the instructor |
| Drive mount (`drive.mount`) | works | not available (the runtime is not a Google VM); use Pegasus storage, `scp`, or GitHub |
| session limit | ~12 h / idle disconnects | your job's `--time` (default 4 h); the tunnel must stay open |
| internet from the kernel | yes | yes (HPC confirmed compute nodes download freely); the class datasets are still pre-tokenized under `$GW1B_SCRATCH/data` |

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
