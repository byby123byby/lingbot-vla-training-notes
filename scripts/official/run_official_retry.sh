#!/usr/bin/env bash
# Run inside the competition robotwin image, after mounting Devzone.
# Stopping this script stops its model server, not the billable cloud instance.
set -euo pipefail
OUT=/workspace/lingbot_retry_20261024
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1
server_pid=
cleanup() {
  result=$?
  trap - EXIT
  if [[ -n "$server_pid" ]]; then
    kill -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  printf '%s\n' "$result" > "$OUT/exit_code.txt"
  date -u +%FT%TZ > "$OUT/finished_utc.txt"
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

date -u +%FT%TZ > "$OUT/started_utc.txt"
cd /RoboTwin
source /opt/robotwin-env/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2:/RoboTwin
export ROBOTWIN_DISABLE_CUROBO=1
export ROBOTWIN_EE_PLANNER=mplib
export PYOPENGL_PLATFORM=egl
{
  df -h /workspace
  git rev-parse HEAD
  git -C XPolicyLab rev-parse HEAD
  git -C experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 rev-parse HEAD
  python - <<'PY'
import json, torch
assert torch.cuda.is_available() and torch.version.hip
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(json.dumps({"torch": torch.__version__, "rocm": torch.version.hip,
                  "gpu_count": torch.cuda.device_count(),
                  "gpu_name": torch.cuda.get_device_name(0),
                  "matrix_shape": list(y.shape)}))
PY
} | tee "$OUT/environment.txt"

setsid timeout 1800 bash experiments/lingbot_vla_v2_6b_robotwin/scripts/launch_official_server.sh \
  0 13400 "$OUT/model_server.log" False > "$OUT/server_launcher.log" 2>&1 &
server_pid=$!
for attempt in {1..120}; do
  if curl -fsS --max-time 2 http://127.0.0.1:13400/healthz > "$OUT/health.json" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    cat "$OUT/server_launcher.log"
    exit 1
  fi
  sleep 5
done
curl -fsS --max-time 5 http://127.0.0.1:13400/healthz > "$OUT/health.json"

timeout 300 python experiments/lingbot_vla_v2_6b_robotwin/scripts/benchmark_official_inference.py \
  --port 13400 --repeats 3 --batch-sizes 1 --output "$OUT/inference.json"

timeout 900 python /workspace/eval_policy_smoke_retry.py \
  --task_name adjust_bottle --task_config demo_clean \
  --policy_name LingBot-VLA-v2 --protocol lingbot_vla_v2 \
  --host 127.0.0.1 --port 13400 --device_id 0 --seed 0 --test_num 1 \
  --expert_check true --accept_expert_info_on_failure true --eval_batch false \
  --additional_info eval_video_log=false 2>&1 | tee "$OUT/evaluation.log"
printf 'SMOKE_TEST_COMPLETED\n'
