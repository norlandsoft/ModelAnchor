#!/usr/bin/env bash
# 停止后台运行的 ModelAnchor
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${MODELANCHOR_PORT:-7990}"
PID_FILE="$ROOT/.run/modelanchor.pid"

if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "==> 停止 pid $PID"
    kill "$PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
else
  echo "没有找到 pid 文件"
fi

# 兜底：按端口找残留进程
LEFTOVER="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
if [ -n "$LEFTOVER" ]; then
  echo "==> 清理端口 $PORT 上的残留进程：$LEFTOVER"
  kill $LEFTOVER 2>/dev/null || true
fi

echo "==> 已停止"
