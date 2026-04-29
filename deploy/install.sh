#!/bin/bash

set -e

echo "========================================"
echo "RKNN INSTALL SCRIPT"
echo "========================================"

if [ "$EUID" -ne 0 ]; then
  echo "Необходим запуск с sudo"
  exit 1
fi

echo "Проверка Python..."
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if [[ "$PYTHON_VERSION" != "3.10" ]]; then
  echo "Требуется Python 3.10.x"
fi

echo "Обновление pip..."
apt install python3-pip
python3 -m pip install --upgrade pip

echo "Поиск requirements.txt..."
REQ_FILE=$(find / -type f -name "requirements.txt" 2>/dev/null | head -n 1)

if [ -n "$REQ_FILE" ]; then
  echo "Найден: $REQ_FILE"
  pip3 install -r "$REQ_FILE"
else
  echo "requirements.txt не найден"
fi

echo "Поиск RKNN .whl..."
WHL_FILE=$(find / -type f -name "rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl" 2>/dev/null | head -n 1)

if [ -n "$WHL_FILE" ]; then
  echo "Найден: $WHL_FILE"
  pip3 install "$WHL_FILE"
else
  echo ".whl файл не найден"
  exit 1
fi

echo "Поиск librknnrt.so..."
SO_FILE=$(find / -type f -name "librknnrt.so" 2>/dev/null | head -n 1)

if [ -z "$SO_FILE" ]; then
  echo "librknnrt.so не найден"
  exit 1
fi

echo "Найден: $SO_FILE"
file "$SO_FILE"

if [ -f "/usr/lib/librknnrt.so" ]; then
  echo "Создание backup..."
  cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak
fi

echo "Установка библиотеки..."
cp -f "$SO_FILE" /usr/lib/librknnrt.so
chmod 755 /usr/lib/librknnrt.so

echo "🔄 Обновление ldconfig..."
ldconfig

if ldconfig -p | grep -q "librknnrt.so"; then
  echo "Библиотека зарегистрирована"
else
  echo "Библиотека не найдена в кеше"
fi

echo "Проверка моделей..."
for file in coatnet_clear.rknn vit_quant.rknn deploy.prototxt res10_300x300_ssd_iter_140000.caffemodel; do
    FOUND=$(find / -type f -name "$file" 2>/dev/null | head -n 1)
    if [ -n "$FOUND" ]; then
        echo "$file найден"
    else
        echo "$file не найден"
    fi
done

echo "========================================"
echo "УСТАНОВКА ЗАВЕРШЕНА"
echo "========================================"