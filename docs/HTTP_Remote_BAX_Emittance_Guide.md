# Remote BAX Emittance Guide

A practical guide to running BAX emittance optimization with an Xopt generator on a separate compute host.

## 1. Architecture

| Component | Location | Responsibility |
| --- | --- | --- |
| Jupyter notebook and Xopt application | **Client Host** | Coordinates the optimization workflow |
| Evaluator | **Client Host** | Evaluates candidate settings and returns observations |
| `CloudpickleHTTPGenerator` client | **Client Host** | Sends generator requests and receives candidates over HTTP |
| Xopt cloudpickle HTTP server | **Remote Compute Backend** | Hosts authenticated generator sessions |
| BAX generator | **Remote Compute Backend** | Fits models and generates candidates using CPU or GPU |

**Data flow:** the Client Host evaluates candidates and sends observations to the Remote Compute Backend. The remote BAX generator uses those observations to propose new candidates, which return to the Client Host for evaluation.

The HTTP generator server and the notebook are separate processes. The cloudpickle endpoint accepts serialized Python objects and can execute code when deserializing them: use it **only with trusted users and trusted code**, with authentication and network isolation. An SSH tunnel protects transport in this development configuration; bearer-token authentication alone does not encrypt HTTP.

## 2. Choose a notebook

| Notebook | Compute mode | Notes |
| --- | --- | --- |
| `examples/http_generator_emittance.ipynb` | CPU | Remote BAX computation without CUDA |
| `examples/http_generator_emittance_gpu.ipynb` | GPU | Remote BAX computation with CUDA |
| `examples/http_generator_emittance_gpu_authenticated.ipynb` | GPU | Explicit authentication-focused GPU example |

**All three examples use bearer-token authentication.** The third is retained as a distinct demonstration of the authenticated GPU workflow.

The notebooks use a local SSH-forwarded URL by default: `http://127.0.0.1:18002`. Set `XOPT_REMOTE_URL` to override it. The token is read from `~/.config/xopt/fedora-http-token` by default; use `XOPT_TOKEN_FILE` to point to a different file. That filename is a testbed legacy default, not a required hostname or a recommended universal name.

## 3. Prepare the Remote Compute Backend

Start the **authenticated cloudpickle Xopt HTTP server** on a trusted Remote Compute Backend. The server must have the compatible Xopt implementation and the BAX/scientific Python dependencies required by the serialized generator. GPU examples require a CUDA-capable backend with working PyTorch/CUDA support.

The server requires `XOPT_HTTP_GENERATOR_TOKEN` to be set before startup. Configure this through your backend's protected environment or secret mechanism; do not print the token, include it in notebooks, or commit it to Git. For the Docker development setup, configure the token in the ignored `docker/deployment.env` file and start the backend using that deployment's documented helper scripts.

For an SSH-tunnel setup, bind the backend's published HTTP port to loopback (`127.0.0.1:8002`) rather than exposing the cloudpickle endpoint on the network. This example assumes port `8002`; adjust it to match the actual deployment.

## 4. Connect the Client Host

The following commands illustrate a **home testbed only**:

| Architecture role | Home-testbed hostname |
| --- | --- |
| Client Host | `lepton` |
| Remote Compute Backend | `fedora` (`10.0.0.87` in this testbed) |

On the Client Host, establish an SSH tunnel in a dedicated terminal:

```bash
ssh -N -L 127.0.0.1:18002:127.0.0.1:8002 \
  ernesto@10.0.0.87
```

Keep that terminal running during notebook execution. In a second Client Host terminal:

```bash
export XOPT_REMOTE_URL=http://127.0.0.1:18002
export XOPT_TOKEN_FILE="$HOME/.config/xopt/fedora-http-token"

# Confirm that the backend is reachable through the tunnel.
curl --noproxy '*' --max-time 5 -sS -o /dev/null \
  -w 'Backend HTTP status: %{http_code}\n' \
  "$XOPT_REMOTE_URL/openapi.json"
```

Expect HTTP `200` for the API description. An unauthenticated request to a protected endpoint should return HTTP `401`:

```bash
curl --noproxy '*' --max-time 5 -sS -o /dev/null \
  -w 'Unauthenticated HTTP status: %{http_code}\n' \
  "$XOPT_REMOTE_URL/sessions/nonexistent/state/cloudpickle"
```

The example token file must already contain the same bearer token configured on the Remote Compute Backend. Protect its filesystem permissions (for example, `chmod 600 "$XOPT_TOKEN_FILE"`). Do not display its contents in logs or terminal transcripts.

**Important:** a pre-existing `XOPT_REMOTE_URL` overrides the notebook default. Check `printf '%s\n' "$XOPT_REMOTE_URL"` if a notebook attempts to contact the backend directly rather than through the SSH tunnel.

## 5. Run an experiment

On the Client Host, activate a Python environment with compatible Xopt, `bax_algorithms`, Jupyter, PyTorch, and the scientific dependencies used by the selected notebook. Open one of the three notebooks and run its cells in order.

The evaluator stays on the Client Host. The remote generator performs model fitting and candidate generation on the Remote Compute Backend. The GPU notebooks request CUDA on the backend; the Client Host does not need a GPU just to run the evaluator and Jupyter interface.

The notebooks load the token from `XOPT_TOKEN_FILE` and pass it to `CloudpickleHTTPGenerator` using `api_key`. For diagnostic plots or local inspection of remote model state, use the notebook's CPU-safe `X.generator.pull()` pattern. Do not replace it with `X.generator.train_model()` as a way to inspect a remote generator; that would request local model training.

## 6. Run regression tests with cell-by-cell progress

From the `bax_algorithms` repository root on the Client Host:

```bash
python scripts/test_http_notebooks.py \
  examples/http_generator_emittance.ipynb \
  examples/http_generator_emittance_gpu.ipynb \
  examples/http_generator_emittance_gpu_authenticated.ipynb
```

The runner prints the first line of each code cell, progress, elapsed time, and a final PASS/FAIL summary. Cells shorter than one second display milliseconds (three decimal places in seconds); longer cells display tenths of a second. A long-running optimization cell reports `RUNNING` until it finishes. Notebook execution happens in memory: the runner does not rewrite the `.ipynb` files.

**Validated home-testbed results (October 2026):**

| Notebook | Result | Time |
| --- | --- | ---: |
| CPU emittance | PASS | 129.3 s |
| GPU emittance | PASS | 75.9 s |
| Authenticated GPU emittance | PASS | 71.2 s |

The authenticated GPU notebook also passed a separate execution with the progress runner (65.9 s). Times are observations from one testbed, not performance guarantees.

## 7. Troubleshooting

| Symptom | Likely cause / action |
| --- | --- |
| Connection refused to `10.0.0.87:8002` | Direct access is blocked intentionally. Set `XOPT_REMOTE_URL=http://127.0.0.1:18002` and use the SSH tunnel. |
| Connection refused to `127.0.0.1:18002` | Tunnel is not running, or the backend server is down. Check the SSH terminal and `/openapi.json`. |
| HTTP `401` | Missing, invalid, or mismatched bearer token. Verify token-file path and backend secret configuration without printing credentials. |
| HTTP `404` for a nonexistent session | Expected when authentication succeeds but that session does not exist. |
| Notebook appears to hang | Model fitting and optimization can be expensive. Use `scripts/test_http_notebooks.py` for per-cell progress. |
| GPU failure | Verify CUDA/PyTorch on the Remote Compute Backend; do not assume the Client Host's CUDA availability is relevant. |
| Wrong backend despite tunnel being healthy | Inspect the exported `XOPT_REMOTE_URL`; environment variables override notebook defaults. |

## 8. Portability beyond the home testbed

For a shared compute host or Kubernetes deployment, keep the same **Client Host / Remote Compute Backend** separation. Replace testbed-specific hostnames, ports, token-file paths, and tunnel setup with the deployment's approved endpoint and authentication method. Do not expose the cloudpickle server directly to untrusted networks; use an authenticated, encrypted, access-controlled transport and isolate the backend execution environment.
