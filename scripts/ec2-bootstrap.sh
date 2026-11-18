#!/usr/bin/env bash
# One-time setup of the EC2 staging box (Ubuntu 24.04, x86, 8 GiB, ap-south-1). Phase 4.2.
# Run as the login user (sudo is used where needed). Safe to re-run.
# After it: log out and back in (docker group), create backend/.env, restore the data dump, run the gate.
set -euo pipefail

SWAP_GB="${SWAP_GB:-4}"
NODE_MAJOR="${NODE_MAJOR:-20}"

echo "== packages"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git make python3 docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo "== node ${NODE_MAJOR} (frontend lint/build in gate 0)"
if ! command -v node >/dev/null || [[ "$(node -v)" != v${NODE_MAJOR}.* ]]; then
  curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | sudo -E bash -
  sudo apt-get install -y nodejs
fi

echo "== uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

echo "== swapfile ${SWAP_GB}G"
if ! sudo swapon --show | grep -q /swapfile; then
  sudo fallocate -l "${SWAP_GB}G" /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi
sudo sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-progenitor.conf >/dev/null

echo "== done"
echo "Next:"
echo "  1. log out and back in (docker group)"
echo "  2. cd backend && cp .env.example .env; set STACK=ec2, TRACING_MODE=cloudwatch, HAPI_JAVA_OPTS=-Xmx3g, LLM_*"
echo "  3. make up; copy hapi.dump + progenitor.dump to data/external/dumps/; make db-restore"
echo "  4. make gate PHASE=<n>   (second SSH session: make gate-status)"
