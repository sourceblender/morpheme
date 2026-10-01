#!/usr/bin/env bash
# Prepare the dedicated benchmark host (Ubuntu, AMD Ryzen 7 9800X3D) that
# runs .github/workflows/benchmark-tracking.yml as a self-hosted GitHub
# Actions runner. See docs/benchmarks.md, "Dedicated host tracking".
#
#   sudo ./scripts/bench_host_setup.sh          # install / update (idempotent)
#   ./scripts/bench_host_setup.sh --check       # report host and runner state
#
# The script never embeds a registration token. It prints the exact
# `config.sh` command the owner runs once, then installs the systemd
# service on the next invocation.
set -euo pipefail

REPO="sourceblender/morpheme"
RUNNER_NAME="bench-9800x3d"
RUNNER_LABELS="bench"
RUNNER_USER="bench"
RUNNER_DIR="/opt/actions-runner"
FIXTURE_CACHE="/var/cache/morpheme/hf-fixtures"
TUNING_SCRIPT="/usr/local/sbin/bench-cpu-tuning"
TUNING_UNIT="/etc/systemd/system/bench-cpu-tuning.service"
APT_PACKAGES=(build-essential curl git python3 jq time ca-certificates util-linux)

log() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- state reporting -------------------------------------------------------

report_cpu() {
  local governors boost epp
  if compgen -G '/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor' > /dev/null; then
    governors=$(cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')
    echo "governor:            ${governors}"
  else
    echo "governor:            (no cpufreq sysfs entries)"
  fi
  if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    boost=$(cat /sys/devices/system/cpu/cpufreq/boost)
    echo "boost (global):      ${boost} (0 = disabled)"
  elif compgen -G '/sys/devices/system/cpu/cpu[0-9]*/cpufreq/boost' > /dev/null; then
    boost=$(cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/boost | sort -u | tr '\n' ' ')
    echo "boost (per cpu):     ${boost}(0 = disabled)"
  else
    echo "boost:               NOT disabled (no boost control exposed; turn Core Performance Boost off in firmware)"
  fi
  if [ -f /sys/devices/system/cpu/amd_pstate/status ]; then
    echo "amd_pstate:          $(cat /sys/devices/system/cpu/amd_pstate/status)"
  fi
  if compgen -G '/sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference' > /dev/null; then
    epp=$(cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')
    echo "energy pref (EPP):   ${epp}"
  fi
  if command -v lscpu > /dev/null; then
    echo "thread siblings of cpus 2-5 (the benchmark cpuset):"
    for cpu in 2 3 4 5; do
      local f="/sys/devices/system/cpu/cpu${cpu}/topology/thread_siblings_list"
      [ -f "$f" ] && echo "  cpu${cpu}: $(cat "$f")"
    done
  fi
}

report_services() {
  local unit
  if [ -f "$TUNING_UNIT" ]; then
    echo "bench-cpu-tuning:    $(systemctl is-enabled bench-cpu-tuning 2> /dev/null || true) / $(systemctl is-active bench-cpu-tuning 2> /dev/null || true)"
  else
    echo "bench-cpu-tuning:    not installed"
  fi
  unit=$(runner_unit_name)
  if [ -n "$unit" ]; then
    echo "runner service:      ${unit} $(systemctl is-enabled "$unit" 2> /dev/null || true) / $(systemctl is-active "$unit" 2> /dev/null || true)"
  elif [ -f "$RUNNER_DIR/.runner" ]; then
    echo "runner service:      configured but service not installed (re-run with sudo)"
  elif [ -d "$RUNNER_DIR" ]; then
    echo "runner service:      runner unpacked in ${RUNNER_DIR} but not configured"
  else
    echo "runner service:      not installed"
  fi
}

report_runner_online() {
  if ! command -v gh > /dev/null; then
    echo "runner online:       unknown (install gh and authenticate, or run: gh api repos/${REPO}/actions/runners)"
    return
  fi
  local status
  if status=$(gh api "repos/${REPO}/actions/runners" --jq ".runners[] | select(.name == \"${RUNNER_NAME}\") | \"\\(.status) busy=\\(.busy) labels=\\([.labels[].name] | join(\",\"))\"" 2> /dev/null); then
    if [ -n "$status" ]; then
      echo "runner online:       ${status}"
    else
      echo "runner online:       ${RUNNER_NAME} is not registered on ${REPO}"
    fi
  else
    echo "runner online:       could not query GitHub (gh auth status?)"
  fi
}

runner_unit_name() {
  # svc.sh names the unit actions.runner.<owner>-<repo>.<name>.service
  local f
  for f in /etc/systemd/system/actions.runner.*.service; do
    [ -e "$f" ] || continue
    basename "$f"
    return
  done
  echo ""
}

check_mode() {
  echo "host:                $(hostname) ($(uname -sr))"
  if [ -r /proc/cpuinfo ]; then
    echo "cpu:                 $(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo)"
  fi
  report_cpu
  report_services
  report_runner_online
  echo "fixture cache:       ${FIXTURE_CACHE} ($(find "$FIXTURE_CACHE" -maxdepth 1 -name '*.json' 2> /dev/null | wc -l | tr -d ' ') files)"
  if [ -x "/home/${RUNNER_USER}/.cargo/bin/rustup" ]; then
    echo "rustup (${RUNNER_USER}):      $(/home/${RUNNER_USER}/.cargo/bin/rustup --version 2> /dev/null | head -1)"
  else
    echo "rustup (${RUNNER_USER}):      not installed"
  fi
  if [ -x /usr/bin/time ]; then
    echo "/usr/bin/time:       present"
  else
    echo "/usr/bin/time:       MISSING (apt install time)"
  fi
}

# --- installation ----------------------------------------------------------

install_packages() {
  log "Installing build dependencies"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends "${APT_PACKAGES[@]}"
}

create_user() {
  if id -u "$RUNNER_USER" > /dev/null 2>&1; then
    log "User ${RUNNER_USER} exists"
  else
    log "Creating user ${RUNNER_USER} (no sudo, no password)"
    useradd --create-home --shell /bin/bash "$RUNNER_USER"
  fi
  install -d -o "$RUNNER_USER" -g "$RUNNER_USER" -m 0755 "$FIXTURE_CACHE"
}

install_rustup() {
  local home="/home/${RUNNER_USER}"
  if [ -x "${home}/.cargo/bin/rustup" ]; then
    log "rustup already installed for ${RUNNER_USER}"
    return
  fi
  log "Installing rustup for ${RUNNER_USER} (minimal profile, stable)"
  sudo -u "$RUNNER_USER" -H bash -c \
    'curl --proto "=https" --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path'
}

install_runner() {
  local release version asset digest url tarball tmp installed
  log "Resolving latest actions/runner release"
  release=$(curl -fsSL -H 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/actions/runner/releases/latest")
  version=$(jq -r '.tag_name | ltrimstr("v")' <<< "$release")
  [ -n "$version" ] && [ "$version" != "null" ] || die "could not resolve the runner version"
  asset="actions-runner-linux-x64-${version}.tar.gz"
  # Prefer the asset digest the API reports; fall back to the sha256 in the release notes.
  digest=$(jq -r --arg a "$asset" '.assets[] | select(.name == $a) | .digest // empty | ltrimstr("sha256:")' <<< "$release")
  if [ -z "$digest" ]; then
    digest=$(jq -r '.body' <<< "$release" | grep -oE 'BEGIN SHA linux-x64 -->[0-9a-f]{64}' | grep -oE '[0-9a-f]{64}' | head -1 || true)
  fi
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || die "no sha256 for ${asset} in the release metadata; refusing to install unverified"
  url=$(jq -r --arg a "$asset" '.assets[] | select(.name == $a) | .browser_download_url' <<< "$release")
  [ -n "$url" ] && [ "$url" != "null" ] || die "release has no asset ${asset}"

  install -d -o "$RUNNER_USER" -g "$RUNNER_USER" -m 0755 "$RUNNER_DIR"
  installed=$(cat "$RUNNER_DIR/.installed-version" 2> /dev/null || true)
  if [ "$installed" = "$version" ] && [ -x "$RUNNER_DIR/bin/Runner.Listener" ]; then
    log "Runner ${version} already unpacked in ${RUNNER_DIR}"
    return
  fi
  if [ -f "$RUNNER_DIR/.runner" ] && [ -n "$installed" ] && [ "$installed" != "$version" ]; then
    # A configured runner self-updates; do not unpack over a live installation.
    log "Runner ${installed} is configured; ${version} will arrive via the runner's self-update"
    return
  fi
  log "Downloading ${asset} (sha256 ${digest:0:12}...)"
  tmp=$(mktemp -d)
  tarball="${tmp}/${asset}"
  curl -fsSL --retry 3 -o "$tarball" "$url"
  echo "${digest}  ${tarball}" | sha256sum -c --quiet - || die "sha256 mismatch for ${asset}"
  tar -xzf "$tarball" -C "$RUNNER_DIR"
  rm -rf "$tmp"
  chown -R "$RUNNER_USER:$RUNNER_USER" "$RUNNER_DIR"
  echo "$version" > "$RUNNER_DIR/.installed-version"
  if [ -x "$RUNNER_DIR/bin/installdependencies.sh" ]; then
    "$RUNNER_DIR/bin/installdependencies.sh" > /dev/null
  fi
  log "Runner ${version} unpacked in ${RUNNER_DIR}"
}

install_cpu_tuning() {
  log "Installing ${TUNING_UNIT}"
  cat > "$TUNING_SCRIPT" <<'EOF'
#!/usr/bin/env bash
# Pin the benchmark host to a steady clock: performance governor on every
# core, turbo/boost off. Each knob is skipped with a message when the
# kernel does not expose it. energy_performance_preference (amd-pstate in
# active mode) is set to performance as well, but it is only a hint: when
# no boost control is writable, boost stays ENABLED and the script says so;
# disable Core Performance Boost in firmware in that case.
set -uo pipefail
cpufreq=/sys/devices/system/cpu/cpufreq
touched=0
for governor in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
  [ -w "$governor" ] || continue
  available="$(dirname "$governor")/scaling_available_governors"
  if [ -r "$available" ] && ! grep -qw performance "$available"; then
    echo "skip: performance governor not available for $governor"
    continue
  fi
  echo performance > "$governor" && touched=$((touched + 1))
done
echo "governor: performance on ${touched} cpus"
if [ -f /sys/devices/system/cpu/amd_pstate/status ]; then
  echo "amd_pstate mode: $(cat /sys/devices/system/cpu/amd_pstate/status)"
fi
eppset=0
for epp in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference; do
  [ -w "$epp" ] || continue
  echo performance > "$epp" 2> /dev/null && eppset=$((eppset + 1))
done
[ "$eppset" -gt 0 ] && echo "energy_performance_preference: performance on ${eppset} cpus (a hint, not a boost control)"
if [ -w "$cpufreq/boost" ]; then
  echo 0 > "$cpufreq/boost" && echo "boost: disabled via $cpufreq/boost"
else
  percpu=0
  for boost in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/boost; do
    [ -w "$boost" ] || continue
    echo 0 > "$boost" 2> /dev/null && percpu=$((percpu + 1))
  done
  if [ "$percpu" -gt 0 ]; then
    echo "boost: disabled on ${percpu} cpus via per-cpu cpufreq/boost"
  else
    echo "boost: NOT disabled; no writable control ($cpufreq/boost or per-cpu cpufreq/boost). Disable Core Performance Boost in firmware for a fixed clock"
  fi
fi
exit 0
EOF
  chmod 0755 "$TUNING_SCRIPT"
  cat > "$TUNING_UNIT" <<EOF
[Unit]
Description=Benchmark host CPU tuning (performance governor, boost off)
After=sysinit.target
ConditionPathExists=/sys/devices/system/cpu

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=${TUNING_SCRIPT}

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --now bench-cpu-tuning.service > /dev/null
  systemctl --no-pager --lines=12 status bench-cpu-tuning.service || true
}

print_config_instructions() {
  cat <<EOF

The runner is unpacked but not yet registered. As the owner, mint a
short-lived registration token (needs repo admin, via an authenticated gh):

    gh api -X POST repos/${REPO}/actions/runners/registration-token -q .token

then configure the runner as the ${RUNNER_USER} user (paste the token for <TOKEN>):

    sudo -iu ${RUNNER_USER} bash -c 'cd ${RUNNER_DIR} && ./config.sh --url https://github.com/${REPO} --token <TOKEN> --labels ${RUNNER_LABELS} --unattended --name ${RUNNER_NAME}'

Finally re-run this script with sudo to install and start the systemd service.
EOF
}

order_runner_after_tuning() {
  # Drop-in so the runner only accepts jobs once the CPU tuning has been applied.
  local unit="$1"
  local dropin="/etc/systemd/system/${unit}.d/10-after-bench-cpu-tuning.conf"
  install -d -m 0755 "$(dirname "$dropin")"
  cat > "$dropin" <<EOF
[Unit]
Wants=bench-cpu-tuning.service
After=bench-cpu-tuning.service
EOF
  systemctl daemon-reload
}

install_service() {
  local unit
  if [ ! -f "$RUNNER_DIR/.runner" ]; then
    print_config_instructions
    return
  fi
  unit=$(runner_unit_name)
  if [ -z "$unit" ]; then
    log "Installing the runner systemd service as ${RUNNER_USER}"
    (cd "$RUNNER_DIR" && ./svc.sh install "$RUNNER_USER")
    unit=$(runner_unit_name)
    [ -n "$unit" ] || die "svc.sh install did not create a systemd unit"
    order_runner_after_tuning "$unit"
    (cd "$RUNNER_DIR" && ./svc.sh start)
  else
    log "Runner service ${unit} already installed"
    order_runner_after_tuning "$unit"
    systemctl is-active --quiet "$unit" || (cd "$RUNNER_DIR" && ./svc.sh start)
  fi
}

main() {
  case "${1:-}" in
    --check)
      check_mode
      ;;
    "")
      [ "$(id -u)" -eq 0 ] || die "run with sudo (or use --check for a read-only report)"
      if ! grep -qsi ubuntu /etc/os-release; then
        warn "this script targets Ubuntu; continuing anyway"
      fi
      install_packages
      create_user
      install_rustup
      install_runner
      install_cpu_tuning
      install_service
      log "Done. Run '$0 --check' to review the host state."
      ;;
    -h | --help)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      ;;
    *)
      die "unknown argument: $1 (use --check or no argument)"
      ;;
  esac
}

main "$@"
