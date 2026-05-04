import os
import json
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("DROWSINESS_DB_PATH", BASE_DIR / "drowsiness_events.db"))
VIDEO_DIR = Path(os.environ.get("DROWSINESS_VIDEO_DIR", BASE_DIR / "drowsy_videos"))
LATEST_FRAME_PATH = Path(os.environ.get("DROWSINESS_LATEST_FRAME_PATH", BASE_DIR / "latest_frame.jpg"))
LIVE_STREAM_PORT = int(os.environ.get("DROWSINESS_LIVE_STREAM_PORT", "8080"))
LIVE_STREAM_PUBLIC_URL = os.environ.get("DROWSINESS_LIVE_STREAM_PUBLIC_URL", "")
LIVE_STREAM_HEALTH_URL = os.environ.get(
    "DROWSINESS_LIVE_STREAM_HEALTH_URL",
    f"http://127.0.0.1:{LIVE_STREAM_PORT}/health"
)


st.set_page_config(
    page_title="Drowsiness dispatcher",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.75rem; }
    .status-ok { color: #16794c; font-weight: 700; }
    .status-warn { color: #a35d00; font-weight: 700; }
    .muted { color: #68707d; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def connect_db():
    if not DB_PATH.exists():
        return None

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=2.0)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=2)
def load_events(limit=200):
    conn = connect_db()

    if conn is None:
        return pd.DataFrame()

    try:
        query = """
            SELECT
                event_uid,
                start_time_local,
                confirmed_time_local,
                end_time_local,
                duration_sec,
                confirmation_sec,
                frame_count,
                fps,
                video_path,
                avg_prob,
                max_prob,
                avg_smooth_prob,
                max_smooth_prob,
                avg_vit_prob,
                avg_coatnet_prob,
                avg_face_conf,
                created_at
            FROM events
            ORDER BY start_time_local DESC
            LIMIT ?
        """
        return pd.read_sql_query(query, conn, params=(limit,))
    finally:
        conn.close()


@st.cache_data(ttl=2)
def load_frame_predictions(event_uid):
    conn = connect_db()

    if conn is None or not event_uid:
        return pd.DataFrame()

    try:
        query = """
            SELECT
                local_timestamp,
                elapsed_sec,
                frame_index,
                status,
                prob,
                smooth_prob,
                vit_prob,
                coatnet_prob,
                face_conf
            FROM frame_predictions
            WHERE event_uid = ?
            ORDER BY frame_index ASC
        """
        return pd.read_sql_query(query, conn, params=(event_uid,))
    finally:
        conn.close()


def parse_dt(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def fmt_dt(value):
    dt = parse_dt(value)

    if dt is None:
        return "n/a"

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def fmt_seconds(value):
    if pd.isna(value):
        return "n/a"

    return f"{float(value):.2f}s"


def fmt_prob(value):
    if pd.isna(value):
        return "n/a"

    return f"{float(value):.3f}"


def video_exists(video_path):
    if not video_path:
        return False, None

    path = Path(video_path)

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.exists(), path


def get_health(events):
    db_status = DB_PATH.exists()
    snapshot_status = LATEST_FRAME_PATH.exists()
    video_dir_status = VIDEO_DIR.exists()
    latest_event_dt = None

    if not events.empty:
        latest_event_dt = parse_dt(events.iloc[0]["start_time_local"])

    return db_status, snapshot_status, video_dir_status, latest_event_dt


@st.cache_data(ttl=2)
def get_live_stream_status():
    try:
        with urllib.request.urlopen(LIVE_STREAM_HEALTH_URL, timeout=1.0) as response:
            body = response.read(512).decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):
        return "offline", ""

    if "status=online" in body:
        return "online", body

    if "status=waiting_for_frame" in body:
        return "waiting", body

    return "unknown", body


def render_sidebar():
    st.sidebar.title("Панель")
    refresh = st.sidebar.checkbox("Автообновление", value=True)
    interval = st.sidebar.slider("Интервал, сек", 2, 30, 5)
    limit = st.sidebar.slider("Событий в истории", 20, 500, 200, step=20)

    if st.sidebar.button("Обновить сейчас", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(f"DB: {DB_PATH}")
    st.sidebar.caption(f"Видео: {VIDEO_DIR}")
    st.sidebar.caption(f"Кадр: {LATEST_FRAME_PATH}")
    st.sidebar.caption(f"MJPEG: :{LIVE_STREAM_PORT}/video")

    return refresh, interval, limit


def render_mjpeg_stream():
    configured_url = LIVE_STREAM_PUBLIC_URL.strip()
    configured_json = json.dumps(configured_url)

    html = f"""
    <div style="width:100%; background:#111827; border-radius:6px; overflow:hidden;">
      <img id="mjpeg-live"
           alt="MJPEG live stream"
           style="display:block; width:100%; height:auto; min-height:260px; object-fit:contain;" />
    </div>
    <script>
      const configuredUrl = {configured_json};
      let src = configuredUrl;

      if (!src) {{
        let protocol = "http:";
        let host = window.location.hostname;

        try {{
          protocol = window.parent.location.protocol || protocol;
          host = window.parent.location.hostname || host;
        }} catch (e) {{}}

        src = protocol + "//" + host + ":{LIVE_STREAM_PORT}/video";
      }}

      const sep = src.includes("?") ? "&" : "?";
      document.getElementById("mjpeg-live").src = src + sep + "t=" + Date.now();
    </script>
    """

    components.html(html, height=420)


def render_live_frame():
    st.subheader("Камера")
    live_status, live_details = get_live_stream_status()

    if live_status == "online":
        render_mjpeg_stream()
        st.caption(f"MJPEG live: {LIVE_STREAM_HEALTH_URL.replace('/health', '/video')}")
        return

    if live_status == "waiting":
        st.info("MJPEG-сервер запущен и ждёт первый обработанный кадр.")
        render_mjpeg_stream()
        return

    if LATEST_FRAME_PATH.exists():
        mtime = datetime.fromtimestamp(LATEST_FRAME_PATH.stat().st_mtime)
        age = datetime.now() - mtime

        st.image(str(LATEST_FRAME_PATH), use_container_width=True)
        st.caption(
            f"MJPEG недоступен, показан fallback-кадр: "
            f"{mtime.strftime('%Y-%m-%d %H:%M:%S')} ({age.seconds}s назад)"
        )
    else:
        st.info(
            "MJPEG-поток и fallback-кадр пока недоступны. "
            "Запусти или проверь основной сервис vit_coatnet."
        )


def render_metrics(events):
    now = datetime.now().astimezone()
    day_start = now - timedelta(hours=24)

    if events.empty:
        today_events = events
    else:
        parsed = events["start_time_local"].apply(parse_dt)
        today_events = events[parsed.apply(lambda dt: dt is not None and dt >= day_start)]

    total = len(events)
    last_24h = len(today_events)
    avg_duration = today_events["duration_sec"].mean() if not today_events.empty else 0
    max_prob = today_events["max_smooth_prob"].max() if not today_events.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего событий", total)
    col2.metric("За 24 часа", last_24h)
    col3.metric("Средняя длительность", fmt_seconds(avg_duration))
    col4.metric("Пик сонливости", fmt_prob(max_prob))


def render_events_table(events):
    st.subheader("События усталости")

    if events.empty:
        if DB_PATH.exists():
            st.info("В базе пока нет подтвержденных событий.")
        else:
            st.warning("База drowsiness_events.db пока не создана основным сервисом.")
        return None

    table = events.copy()
    table["start"] = table["start_time_local"].apply(fmt_dt)
    table["end"] = table["end_time_local"].apply(fmt_dt)
    table["duration"] = table["duration_sec"].apply(fmt_seconds)
    table["avg"] = table["avg_smooth_prob"].apply(fmt_prob)
    table["max"] = table["max_smooth_prob"].apply(fmt_prob)
    table["face"] = table["avg_face_conf"].apply(fmt_prob)
    table["video"] = table["video_path"].apply(lambda path: "yes" if video_exists(path)[0] else "missing")

    display = table[
        [
            "event_uid",
            "start",
            "end",
            "duration",
            "frame_count",
            "avg",
            "max",
            "face",
            "video",
        ]
    ]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "event_uid": st.column_config.TextColumn("UID", width="medium"),
            "start": "Начало",
            "end": "Конец",
            "duration": "Длительность",
            "frame_count": "Кадры",
            "avg": "Средн.",
            "max": "Макс.",
            "face": "Лицо",
            "video": "Видео",
        },
    )

    options = events["event_uid"].tolist()
    return st.selectbox("Открыть событие", options=options, format_func=lambda uid: uid[:8])


def render_event_detail(events, event_uid):
    if not event_uid:
        return

    event = events[events["event_uid"] == event_uid].iloc[0]

    st.subheader(f"Детали события {event_uid[:8]}")

    left, right = st.columns([1.1, 1])

    with left:
        exists, path = video_exists(event["video_path"])

        if exists:
            st.video(str(path))
            st.caption(str(path))
        else:
            st.warning(f"Видео не найдено: {event['video_path']}")

    with right:
        st.write(
            {
                "Начало": fmt_dt(event["start_time_local"]),
                "Подтверждено": fmt_dt(event["confirmed_time_local"]),
                "Конец": fmt_dt(event["end_time_local"]),
                "Длительность": fmt_seconds(event["duration_sec"]),
                "Кадров": int(event["frame_count"]),
                "FPS записи": float(event["fps"]),
                "Средняя вероятность": fmt_prob(event["avg_smooth_prob"]),
                "Максимальная вероятность": fmt_prob(event["max_smooth_prob"]),
                "ViT средн.": fmt_prob(event["avg_vit_prob"]),
                "CoAtNet средн.": fmt_prob(event["avg_coatnet_prob"]),
                "Face confidence средн.": fmt_prob(event["avg_face_conf"]),
            }
        )

    frames = load_frame_predictions(event_uid)

    if not frames.empty:
        st.line_chart(
            frames.set_index("elapsed_sec")[["smooth_prob", "vit_prob", "coatnet_prob"]],
            height=260,
        )

        with st.expander("Покадровые предсказания"):
            st.dataframe(frames, use_container_width=True, hide_index=True)


def main():
    refresh, interval, limit = render_sidebar()
    events = load_events(limit=limit)

    st.title("Диспетчер усталости")

    db_status, snapshot_status, video_dir_status, latest_event_dt = get_health(events)
    live_status, _ = get_live_stream_status()
    status_cols = st.columns(5)
    status_cols[0].markdown(
        f"DB: <span class='{'status-ok' if db_status else 'status-warn'}'>"
        f"{'online' if db_status else 'missing'}</span>",
        unsafe_allow_html=True,
    )
    status_cols[1].markdown(
        f"Кадр: <span class='{'status-ok' if snapshot_status else 'status-warn'}'>"
        f"{'online' if snapshot_status else 'waiting'}</span>",
        unsafe_allow_html=True,
    )
    status_cols[2].markdown(
        f"Видео: <span class='{'status-ok' if video_dir_status else 'status-warn'}'>"
        f"{'online' if video_dir_status else 'missing'}</span>",
        unsafe_allow_html=True,
    )
    status_cols[3].markdown(
        f"MJPEG: <span class='{'status-ok' if live_status == 'online' else 'status-warn'}'>"
        f"{live_status}</span>",
        unsafe_allow_html=True,
    )
    status_cols[4].markdown(
        f"Последнее событие: <span class='muted'>"
        f"{latest_event_dt.strftime('%Y-%m-%d %H:%M:%S') if latest_event_dt else 'n/a'}</span>",
        unsafe_allow_html=True,
    )

    render_metrics(events)

    live_col, events_col = st.columns([1, 1.25])

    with live_col:
        render_live_frame()

    with events_col:
        selected_uid = render_events_table(events)

    render_event_detail(events, selected_uid)

    if refresh:
        time.sleep(interval)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
