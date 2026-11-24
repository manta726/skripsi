from datetime import datetime, timedelta

FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_waktu(teks):
    """'2026-10-29 08:57:38' -> datetime. Kosong atau salah format -> None."""
    if teks is None:
        return None
    t = str(teks).strip()
    if not t or t.lower() in ("nan", "nat", "none"):
        return None
    try:
        return datetime.strptime(t, FORMAT)
    except ValueError:
        return None


def jam_crossing(mulai, frame, fps, frame_pertama=1):
    """mulai: datetime (jam di frame pertama). frame: nomor Frame crossing (mulai dari 1)."""
    if mulai is None or frame is None or not fps or fps <= 0:
        return None
    return mulai + timedelta(seconds=(frame - frame_pertama) / fps)


def format_waktu(dt):
    return dt.strftime(FORMAT) if dt else ""


def selisih_detik(dt_hitung, dt_aktual):
    if dt_hitung is None or dt_aktual is None:
        return None
    return (dt_hitung - dt_aktual).total_seconds()