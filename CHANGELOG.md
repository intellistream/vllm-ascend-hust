# Changelog

## Unreleased

### Changed

- Added optional SSH-over-443 checkout support to the trusted benchmark
  workflow. When the `VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY` secret is set,
  benchmark repository fetches use `ssh.github.com:443`; otherwise the workflow
  keeps the default HTTPS checkout path.
- Added the sibling `vllm-hust-website` checkout to the trusted benchmark
  workflow so preview aggregation can complete instead of failing during
  post-processing.