#!/usr/bin/env bash
# Install or update the NovaWealth voice agent on an Ubuntu 24.04 server
# (AWS Lightsail or EC2) and serve it over HTTPS with Caddy.
#
# On the server:
#   curl -fsSL https://raw.githubusercontent.com/molshreepandey/nova-ai-call/main/deploy/setup.sh -o setup.sh
#   sudo bash setup.sh
#
# It asks for the Groq API key (first run only) and an optional access code.
# Optional environment overrides:
#   DOMAIN=voice.example.com   your own domain (default: <public-ip>.sslip.io)
#   ACCESS_CODE=...            set/replace the access code without a prompt
#   BRANCH=main                git branch to deploy
# Re-run the same command any time to pull the latest code and restart.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/molshreepandey/nova-ai-call.git}"
BRANCH="${BRANCH:-main}"
APP_DIR=/opt/nova-ai-call
APP_USER=nova
SERVICE=nova-agent
TORCH_VERSION=2.14.1   # CPU build; must match the version requirements resolve to

log() { printf '\n\033[1;33m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run with sudo: sudo bash setup.sh"
. /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "this script supports Ubuntu (found ${ID:-unknown})"

# ---------------------------------------------------------------- swap
mem_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
if (( mem_mb < 3000 )) && ! swapon --show | grep -q /swapfile; then
  log "Adding a 2 GB swap file (server has ${mem_mb} MB RAM)"
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# ---------------------------------------------------------------- packages
log "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl \
  debian-keyring debian-archive-keyring apt-transport-https gnupg >/dev/null
if ! command -v caddy >/dev/null; then
  log "Installing Caddy (automatic HTTPS)"
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq && apt-get install -y -qq caddy >/dev/null
fi

# ---------------------------------------------------------------- code
id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
if [[ -d "$APP_DIR/.git" ]]; then
  log "Updating code ($BRANCH)"
  sudo -u "$APP_USER" git -C "$APP_DIR" fetch -q origin "$BRANCH"
  sudo -u "$APP_USER" git -C "$APP_DIR" checkout -q "$BRANCH"
  sudo -u "$APP_USER" git -C "$APP_DIR" reset -q --hard "origin/$BRANCH"
else
  log "Cloning $REPO_URL ($BRANCH)"
  install -d -o "$APP_USER" -g "$APP_USER" "$APP_DIR"
  sudo -u "$APP_USER" git clone -q --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

# ---------------------------------------------------------------- python env
log "Installing Python dependencies (a few minutes on first run)"
as_app() { sudo -u "$APP_USER" -H bash -c "cd '$APP_DIR' && $*"; }
[[ -x "$APP_DIR/.venv/bin/python" ]] || as_app "python3 -m venv .venv"
as_app ".venv/bin/pip install -q --upgrade pip"
# CPU-only torch first: the default Linux wheel pulls ~3 GB of CUDA libraries a server doesn't need.
as_app ".venv/bin/pip install -q torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cpu"
as_app ".venv/bin/pip install -q -r requirements.txt"

# ---------------------------------------------------------------- configuration
env_file="$APP_DIR/.env"
fresh_env=false
if [[ ! -f "$env_file" ]]; then
  cp "$APP_DIR/.env.example" "$env_file"
  fresh_env=true
fi
set_env() {  # set_env KEY VALUE: replace or append KEY=VALUE in .env
  local key=$1 value=$2
  if grep -q "^${key}=" "$env_file"; then
    python3 - "$env_file" "$key" "$value" <<'PY'
import sys
path, key, value = sys.argv[1:]
lines = open(path).read().splitlines()
lines = [f"{key}={value}" if line.startswith(f"{key}=") else line for line in lines]
open(path, "w").write("\n".join(lines) + "\n")
PY
  else
    printf '%s=%s\n' "$key" "$value" >> "$env_file"
  fi
}

if ! grep -q '^GROQ_API_KEY=gsk_' "$env_file"; then
  key="${GROQ_API_KEY:-}"
  [[ -n "$key" ]] || read -rsp "Groq API key (input hidden): " key </dev/tty && echo
  [[ "$key" == gsk_* ]] || die "that doesn't look like a Groq key (should start with gsk_)"
  set_env GROQ_API_KEY "$key"
fi
if [[ -n "${ACCESS_CODE+x}" ]]; then
  set_env ACCESS_CODE "$ACCESS_CODE"
elif $fresh_env || ! grep -q '^ACCESS_CODE=' "$env_file"; then
  read -rp "Access code to protect the public site (Enter for none): " code </dev/tty
  set_env ACCESS_CODE "$code"
fi
set_env HOST 127.0.0.1
set_env PORT 8000
chown "$APP_USER:$APP_USER" "$env_file" && chmod 600 "$env_file"

# ---------------------------------------------------------------- knowledge base
log "Building the knowledge base"
as_app ".venv/bin/python -m rag.ingest" | tail -1

# ---------------------------------------------------------------- service
log "Starting the app as a service"
cat > /etc/systemd/system/${SERVICE}.service <<UNIT
[Unit]
Description=NovaWealth voice agent (FastAPI)
After=network-online.target
Wants=network-online.target

[Service]
User=${APP_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/python -m web.app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable -q "$SERVICE"
systemctl restart "$SERVICE"

for _ in $(seq 1 90); do
  curl -fs -o /dev/null http://127.0.0.1:8000/ && break
  sleep 2
done
curl -fs -o /dev/null http://127.0.0.1:8000/ || die "app did not start; see: journalctl -u $SERVICE -n 50"

# ---------------------------------------------------------------- HTTPS
public_ip=$(curl -fs https://checkip.amazonaws.com | tr -d '[:space:]')
domain="${DOMAIN:-${public_ip//./-}.sslip.io}"
log "Configuring HTTPS for $domain"
cat > /etc/caddy/Caddyfile <<CADDY
${domain} {
    encode gzip
    reverse_proxy 127.0.0.1:8000
}
CADDY
systemctl reload caddy 2>/dev/null || systemctl restart caddy

log "Done"
echo "  Open:     https://${domain}"
echo "  Logs:     sudo journalctl -u ${SERVICE} -f"
echo "  Update:   sudo bash setup.sh   (pulls the latest code and restarts)"
echo "  The first HTTPS request can take ~30 s while the certificate is issued."
echo "  Ports 80 and 443 must be open in the AWS firewall / security group."
