#!/bin/bash

set -e

if [ "$EUID" -ne 0 ]; then
  echo "[ERROR] Нужно запустить с sudo"
  echo "Пример:"
  echo "  sudo bash install_dashboard_service.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="drowsiness_dashboard"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
PYTHON_PATH="$(command -v python3)"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
DASHBOARD_HOST="${DASHBOARD_HOST:-0.0.0.0}"
LIVE_STREAM_PORT="${LIVE_STREAM_PORT:-8080}"
DB_PATH="$SCRIPT_DIR/drowsiness_events.db"
VIDEO_DIR="$SCRIPT_DIR/drowsy_videos"
LATEST_FRAME_PATH="$SCRIPT_DIR/latest_frame.jpg"

if [ -z "$PYTHON_PATH" ]; then
  echo "[ERROR] python3 не найден"
  exit 1
fi

if [ ! -f "$SCRIPT_DIR/dispatcher_dashboard.py" ]; then
  echo "[ERROR] dispatcher_dashboard.py не найден"
  exit 1
fi

"$PYTHON_PATH" -m pip install -r "$SCRIPT_DIR/dashboard_requirements.txt" --ignore-installed

if ! "$PYTHON_PATH" -m streamlit --version >/dev/null 2>&1; then
  echo "[ERROR] streamlit не запускается как Python-модуль"
  exit 1
fi

mkdir -p "$VIDEO_DIR"
chmod 755 "$SCRIPT_DIR" "$VIDEO_DIR"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Drowsiness Dispatcher Dashboard
After=network.target vit_coatnet.service
Wants=vit_coatnet.service

[Service]
Type=simple
User=root
WorkingDirectory=$SCRIPT_DIR
Environment=PYTHONUNBUFFERED=1
Environment=DROWSINESS_DB_PATH=$DB_PATH
Environment=DROWSINESS_VIDEO_DIR=$VIDEO_DIR
Environment=DROWSINESS_LATEST_FRAME_PATH=$LATEST_FRAME_PATH
Environment=DROWSINESS_LIVE_STREAM_PORT=$LIVE_STREAM_PORT
Environment=DROWSINESS_LIVE_STREAM_HEALTH_URL=http://127.0.0.1:$LIVE_STREAM_PORT/health
Environment=STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ExecStart=$PYTHON_PATH -m streamlit run $SCRIPT_DIR/dispatcher_dashboard.py --server.address $DASHBOARD_HOST --server.port $DASHBOARD_PORT --server.headless true
Restart=always
RestartSec=3

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo "[OK] Dashboard service started"
echo "URL локально:  http://127.0.0.1:$DASHBOARD_PORT"
echo "URL в сети:    http://<IP_устройства>:$DASHBOARD_PORT"
echo "MJPEG stream: http://<IP_устройства>:$LIVE_STREAM_PORT/video"
echo ""
echo "Статус:"
echo "  systemctl status $SERVICE_NAME --no-pager"
echo ""
echo "Логи:"
echo "  journalctl -u $SERVICE_NAME -f"
