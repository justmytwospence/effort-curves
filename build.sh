#!/usr/bin/env bash
# Build the manylinux wheel in the pinned maturin container (no Rust toolchain on the host), for
# x86-64-v3 CPUs (AVX2). TARGET_CPU=native tunes it to this machine instead.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ~/.cache/effort-curves-cargo
docker run --rm -v "$PWD":/io -w /io -v ~/.cache/effort-curves-cargo:/root/.cargo/registry \
  -e RUSTFLAGS="-C target-cpu=${TARGET_CPU:-x86-64-v3}" --entrypoint bash ghcr.io/pyo3/maturin:v1.9.6 -c \
  "maturin build --release -o dist 2>&1; rc=\$?; chown -R $(id -u):$(id -g) target dist Cargo.lock 2>/dev/null; exit \$rc"
