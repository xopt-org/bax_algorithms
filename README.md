# bax_algorithms

Algorithms for Bayesian Algorithm Execution (BAX) with [Xopt](https://github.com/ChristopherMayes/Xopt).

## Remote BAX emittance examples

These examples separate **experiment evaluation** from **generator computation**:

- **Client Host:** runs Jupyter, the Xopt application, and the evaluator.
- **Remote Compute Backend:** runs the Xopt HTTP server and performs BAX model fitting and candidate generation on CPU or GPU.

The client connects to the backend through an authenticated HTTP generator session. For the development testbed, HTTP traffic is carried through an SSH tunnel.

| Notebook | Remote generator compute | Authentication |
| --- | --- | --- |
| [`examples/http_generator_emittance.ipynb`](examples/http_generator_emittance.ipynb) | CPU | Bearer token |
| [`examples/http_generator_emittance_gpu.ipynb`](examples/http_generator_emittance_gpu.ipynb) | GPU | Bearer token |
| [`examples/http_generator_emittance_gpu_authenticated.ipynb`](examples/http_generator_emittance_gpu_authenticated.ipynb) | GPU | Bearer token (explicit example) |

**Quick start:** configure access to a running Remote Compute Backend, establish an SSH tunnel, set `XOPT_REMOTE_URL`, and run one of the notebooks from a Python environment with Xopt and the example dependencies installed. Credentials are loaded from a local file, never from notebook source.

See **[Remote BAX Emittance Guide](docs/HTTP_Remote_BAX_Emittance_Guide.md)** for setup, commands, regression testing, and troubleshooting.

## Regression testing

Run notebook execution tests from the repository root:

```bash
python scripts/test_http_notebooks.py \
  examples/http_generator_emittance.ipynb \
  examples/http_generator_emittance_gpu.ipynb \
  examples/http_generator_emittance_gpu_authenticated.ipynb
```

The runner reports per-cell progress and timings and does not overwrite the notebook files. It requires a working Remote Compute Backend, SSH tunnel, and authentication token; these are integration tests, not offline unit tests.
