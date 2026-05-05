#!/bin/bash

set -e

echo "========================================"
echo "RKNN INSTALL SCRIPT"
echo "========================================"

if [ "$EUID" -ne 0 ]; then
  echo "[ERROR] Необходим запуск с sudo"
  echo "Пример:"
  echo "  sudo bash install.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICE_NAME="vit_coatnet"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

PY_SCRIPT="$SCRIPT_DIR/vit_coatnet_hybrid.py"

VIT_MODEL="$SCRIPT_DIR/vit_quant.rknn"
COATNET_MODEL="$SCRIPT_DIR/coatnet_clear.rknn"

FACE_PROTO="$SCRIPT_DIR/deploy.prototxt"
FACE_MODEL="$SCRIPT_DIR/res10_300x300_ssd_iter_140000.caffemodel"

WHL_FILE="$SCRIPT_DIR/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"

REQ_FILE="$SCRIPT_DIR/requirements.txt"

echo "[INFO] Рабочая папка: $SCRIPT_DIR"

echo "========================================"
echo "Проверка файлов проекта"
echo "========================================"

if [ ! -f "$PY_SCRIPT" ]; then
  echo "[ERROR] Не найден Python-скрипт:"
  echo "  $PY_SCRIPT"
  exit 1
fi

if [ ! -f "$VIT_MODEL" ]; then
  echo "[ERROR] Не найдена модель:"
  echo "  $VIT_MODEL"
  exit 1
fi

if [ ! -f "$COATNET_MODEL" ]; then
  echo "[ERROR] Не найдена модель:"
  echo "  $COATNET_MODEL"
  exit 1
fi

if [ ! -f "$FACE_PROTO" ]; then
  echo "[ERROR] Не найден файл:"
  echo "  $FACE_PROTO"
  exit 1
fi

if [ ! -f "$FACE_MODEL" ]; then
  echo "[ERROR] Не найден файл:"
  echo "  $FACE_MODEL"
  exit 1
fi

if [ ! -f "$WHL_FILE" ]; then
  echo "[ERROR] Не найден RKNN wheel-файл:"
  echo "  $WHL_FILE"
  exit 1
fi

echo "[OK] Все основные файлы найдены"

echo "========================================"
echo "Проверка Python"
echo "========================================"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 не найден"
  exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

echo "[INFO] Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.10" ]]; then
  echo "[ERROR] Требуется Python 3.10.x"
  echo "Текущая версия: $PYTHON_VERSION"
  echo ""
  echo "RKNN wheel у тебя собран под cp310, то есть под Python 3.10."
  echo "Файл:"
  echo "  $(basename "$WHL_FILE")"
  exit 1
fi

PYTHON_PATH=$(which python3)

echo "[INFO] Python path: $PYTHON_PATH"

echo "========================================"
echo "Установка системных зависимостей"
echo "========================================"

apt update

apt install -y \
  python3-pip \
  python3-dev \
  python3-venv \
  sqlite3 \
  ffmpeg \
  v4l-utils \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1

echo "========================================"
echo "Обновление pip"
echo "========================================"

python3 -m pip install --upgrade pip setuptools wheel

echo "========================================"
echo "Установка Python-зависимостей"
echo "========================================"

if [ -f "$REQ_FILE" ]; then
  echo "[INFO] Найден requirements.txt:"
  echo "  $REQ_FILE"
  python3 -m pip install -r "$REQ_FILE"
else
  echo "[WARN] requirements.txt не найден рядом со скриптом"
  echo "[INFO] Устанавливаю базовые зависимости вручную"

  python3 -m pip install \
    numpy \
    opencv-python
fi

echo "========================================"
echo "Установка RKNN Toolkit Lite2"
echo "========================================"

echo "[INFO] Wheel:"
echo "  $WHL_FILE"

python3 -m pip install --force-reinstall "$WHL_FILE"

echo "========================================"
echo "Проверка импорта Python-модулей"
echo "========================================"

python3 - <<'PY_CHECK'
import sys

print("[CHECK] Python:", sys.version)

try:
    import cv2
    print("[OK] cv2:", cv2.__version__)
except Exception as e:
    print("[ERROR] cv2 import failed:", e)
    raise

try:
    import numpy as np
    print("[OK] numpy:", np.__version__)
except Exception as e:
    print("[ERROR] numpy import failed:", e)
    raise

try:
    from rknnlite.api import RKNNLite
    print("[OK] rknnlite imported")
except Exception as e:
    print("[ERROR] rknnlite import failed:", e)
    raise

try:
    import sqlite3
    print("[OK] sqlite3 imported")
except Exception as e:
    print("[ERROR] sqlite3 import failed:", e)
    raise
PY_CHECK

echo "========================================"
echo "Установка librknnrt.so"
echo "========================================"

LOCAL_SO="$SCRIPT_DIR/librknnrt.so"

if [ -f "$LOCAL_SO" ]; then
  SO_FILE="$LOCAL_SO"
else
  SO_FILE=$(find "$SCRIPT_DIR" /usr /lib /opt -type f -name "librknnrt.so" 2>/dev/null | head -n 1)
fi

if [ -z "$SO_FILE" ]; then
  echo "[ERROR] librknnrt.so не найден"
  echo "Положи librknnrt.so рядом с install.sh или установи RKNN runtime отдельно."
  exit 1
fi

echo "[INFO] Найден:"
echo "  $SO_FILE"

file "$SO_FILE"

if [ -f "/usr/lib/librknnrt.so" ]; then
  BACKUP_PATH="/usr/lib/librknnrt.so.bak.$(date +%Y%m%d_%H%M%S)"
  echo "[INFO] Создаю backup:"
  echo "  $BACKUP_PATH"
  cp /usr/lib/librknnrt.so "$BACKUP_PATH"
fi

cp -f "$SO_FILE" /usr/lib/librknnrt.so
chmod 755 /usr/lib/librknnrt.so

ldconfig

if ldconfig -p | grep -q "librknnrt.so"; then
  echo "[OK] librknnrt.so зарегистрирована"
else
  echo "[WARN] librknnrt.so не найдена в кеше ldconfig"
fi

echo "========================================"
echo "Проверка камеры"
echo "========================================"

if [ -e "/dev/video0" ]; then
  echo "[OK] Найдена камера /dev/video0"
  v4l2-ctl --device=/dev/video0 --info || true
else
  echo "[WARN] /dev/video0 не найдена"
  echo "Проверь камеру командой:"
  echo "  ls -l /dev/video*"
fi

echo "========================================"
echo "Создание папок для данных"
echo "========================================"

mkdir -p "$SCRIPT_DIR/drowsy_videos"

chmod 755 "$SCRIPT_DIR"
chmod 755 "$SCRIPT_DIR/drowsy_videos"

echo "========================================"
echo "Проверка файлов моделей"
echo "========================================"

for file in \
  "$VIT_MODEL" \
  "$COATNET_MODEL" \
  "$FACE_PROTO" \
  "$FACE_MODEL"
do
  if [ -f "$file" ]; then
    echo "[OK] $(basename "$file") найден, размер: $(du -h "$file" | awk '{print $1}')"
  else
    echo "[ERROR] $(basename "$file") не найден"
    exit 1
  fi
done

FACE_MODEL_SIZE=$(stat -c%s "$FACE_MODEL")

if [ "$FACE_MODEL_SIZE" -lt 5000000 ]; then
  echo "[ERROR] Face model слишком маленькая:"
  echo "  $FACE_MODEL_SIZE bytes"
  echo "Скорее всего файл повреждён:"
  echo "  $FACE_MODEL"
  exit 1
fi

echo "========================================"
echo "Создание systemd-сервиса"
echo "========================================"

if systemctl is-active --quiet "$SERVICE_NAME"; then
  echo "[INFO] Останавливаю старый сервис"
  systemctl stop "$SERVICE_NAME"
fi

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=ViT + CoAtNet Drowsiness Detection
After=multi-user.target

[Service]
Type=simple
User=root
WorkingDirectory=$SCRIPT_DIR
Environment=PYTHONUNBUFFERED=1
Environment=SHOW_WINDOW=0
Environment=LIVE_STREAM_ENABLED=1
Environment=LIVE_STREAM_HOST=0.0.0.0
Environment=LIVE_STREAM_PORT=8080
Environment=LIVE_STREAM_MAX_FPS=8
ExecStart=$PYTHON_PATH $PY_SCRIPT
Restart=always
RestartSec=3

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "[INFO] Создан сервис:"
echo "  $SERVICE_FILE"

echo "========================================"
echo "Перезагрузка systemd"
echo "========================================"

systemctl daemon-reload

echo "========================================"
echo "Включение автозапуска"
echo "========================================"

systemctl enable "$SERVICE_NAME"

echo "========================================"
echo "Запуск сервиса"
echo "========================================"

systemctl restart "$SERVICE_NAME"

sleep 2

echo "========================================"
echo "Статус сервиса"
echo "========================================"

systemctl status "$SERVICE_NAME" --no-pager || true

echo "========================================"
echo "УСТАНОВКА ЗАВЕРШЕНА"
echo "========================================"

echo ""
echo "Полезные команды:"
echo ""
echo "  Статус сервиса:"
echo "    systemctl status $SERVICE_NAME --no-pager"
echo ""
echo "  Логи сервиса:"
echo "    journalctl -u $SERVICE_NAME -f"
echo ""
echo "  Остановить сервис:"
echo "    systemctl stop $SERVICE_NAME"
echo ""
echo "  Запустить сервис:"
echo "    systemctl start $SERVICE_NAME"
echo ""
echo "  Перезапустить сервис:"
echo "    systemctl restart $SERVICE_NAME"
echo ""
echo "  Посмотреть события в базе:"
echo "    sqlite3 $SCRIPT_DIR/drowsiness_events.db \"SELECT COUNT(*) FROM events;\""
echo ""
echo "  Посмотреть видео:"
echo "    ls -lh $SCRIPT_DIR/drowsy_videos"
echo ""
