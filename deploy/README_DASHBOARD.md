# Drowsiness dispatcher dashboard

Веб-панель для диспетчера запускается отдельным сервисом и читает данные,
которые создает основной сервис `vit_coatnet`.

## Что показывает

- последний обработанный кадр из `latest_frame.jpg`;
- MJPEG-live поток обработанных кадров из `vit_coatnet`;
- статус наличия базы, кадра и каталога видео;
- статистику событий усталости за последние 24 часа;
- таблицу подтвержденных событий из `drowsiness_events.db`;
- видео каждого события: оригинальный AVI хранится в `drowsy_videos`,
  а dashboard готовит MP4-копию в `drowsy_videos/web_videos` для браузера;
- график `smooth_prob`, `vit_prob`, `coatnet_prob` по кадрам.

## Установка на устройстве

```bash
cd ./test1/deploy
sudo bash install.sh
sudo bash install_dashboard_service.sh
```

После установки открой:

```text
http://<IP_устройства>:8501
```

MJPEG-поток основного сервиса доступен отдельно:

```text
http://<IP_устройства>:8080/video
```

## Проверка

```bash
systemctl status vit_coatnet --no-pager
systemctl status drowsiness_dashboard --no-pager
journalctl -u vit_coatnet -f
journalctl -u drowsiness_dashboard -f
```

Если нужен другой порт:

```bash
sudo DASHBOARD_PORT=8600 bash install_dashboard_service.sh
```

Если нужен другой MJPEG-порт для основного сервиса, поменяй `LIVE_STREAM_PORT`
в `install.sh` перед установкой или в systemd unit после установки.

## Файлы данных

- `drowsiness_events.db` создается автоматически основным сервисом.
- `latest_frame.jpg` обновляется основным сервисом примерно два раза в секунду.
- `drowsy_videos/` создается установщиком и заполняется реальными эпизодами.
- `drowsy_videos/web_videos/` содержит MP4-копии для просмотра в браузере.
- `:8080/video` отдает MJPEG-поток без повторного открытия камеры.
