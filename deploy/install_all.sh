#!/bin/bash

set -e

echo "========================================"
echo "DROWSINESS FULL INSTALL"
echo "========================================"

if [ "$EUID" -ne 0 ]; then
  echo "[ERROR] Нужно запустить с sudo"
  echo "Пример:"
  echo "  sudo bash install_all.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAIN_INSTALL="$SCRIPT_DIR/install.sh"
DASHBOARD_INSTALL="$SCRIPT_DIR/install_dashboard_service.sh"

if [ ! -f "$MAIN_INSTALL" ]; then
  echo "[ERROR] Не найден основной установщик:"
  echo "  $MAIN_INSTALL"
  exit 1
fi

if [ ! -f "$DASHBOARD_INSTALL" ]; then
  echo "[ERROR] Не найден установщик dashboard:"
  echo "  $DASHBOARD_INSTALL"
  exit 1
fi

echo "========================================"
echo "1/2 Установка основного сервиса vit_coatnet"
echo "========================================"

bash "$MAIN_INSTALL"

echo "========================================"
echo "2/2 Установка dashboard-сервиса"
echo "========================================"

bash "$DASHBOARD_INSTALL"

echo "========================================"
echo "УСТАНОВКА ЗАВЕРШЕНА"
echo "========================================"
echo ""
echo "Проверка сервисов:"
echo "  systemctl status vit_coatnet --no-pager"
echo "  systemctl status drowsiness_dashboard --no-pager"
echo ""
echo "Логи:"
echo "  journalctl -u vit_coatnet -f"
echo "  journalctl -u drowsiness_dashboard -f"
echo ""
echo "Dashboard:"
echo "  http://<IP_устройства>:8501"
