#!/usr/bin/env bash
# Build the wheel in the pinned maturin container (no Rust toolchain on the host), tuned for this CPU.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ~/.cache/effort-curves-cargo
docker run --rm -v "$PWD":/io -w /io -v ~/.cache/effort-curves-cargo:/root/.cargo/registry \
  -e RUSTFLAGS="-C target-cpu=native" --entrypoint bash ghcr.io/pyo3/maturin:v1.9.6 -c \
  "maturin build --release -o dist --compatibility off 2>&1; rc=\$?; chown -R $(id -u):$(id -g) target dist Cargo.lock 2>/dev/null; exit \$rc"
