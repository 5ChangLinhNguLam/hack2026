"""Đưa thời tiết của HAI nguồn về CÙNG một vector số.

Vì sao cần: Practice/Hackathon_Dataset_Redacted ghi thời tiết bằng đúng bộ tham
số `carla.WeatherParameters` (cloudiness, precipitation, wetness, fog_density,
sun_altitude_angle — thang 0..100 và độ). DeepAccident chỉ ghi TÊN PRESET ở
token đầu file meta (`ClearNoon`, `HardRainNight`, ...). Không quy về một hệ số
thì feature thời tiết chỉ là hai không gian rời nhau và model học được đúng
"đây là dataset nào" — tức một cái shortcut domain, tệ hơn là không có gì.

Bảng dưới suy thẳng từ preset của CARLA, và PHÂN TÍCH TỪ TÊN chứ không tra bảng
20 dòng: dữ liệu thật có cả `MidRainyNoon` lẫn `MidRainSunset` (khác chính tả),
tra bảng cứng là thiếu đúng những token viết lệch.

Đo được trên 5 gói: 20 token, phủ hết bởi 7 tiền tố mưa/mây × 3 hậu tố thời điểm.
"""

# tiền tố -> (cloudiness, precipitation, wetness, fog_density)
# thứ tự kiểm tra QUAN TRỌNG: "WetCloudy" phải khớp trước "Wet" và "Cloudy".
_SKY = [
    ("WetCloudy", (80.0, 0.0, 50.0, 0.0)),
    ("HardRain", (90.0, 60.0, 100.0, 7.0)),
    ("MidRainy", (80.0, 30.0, 100.0, 3.0)),
    ("MidRain", (80.0, 30.0, 100.0, 3.0)),
    ("SoftRain", (70.0, 15.0, 50.0, 0.0)),
    ("Cloudy", (80.0, 0.0, 0.0, 0.0)),
    ("Clear", (15.0, 0.0, 0.0, 0.0)),
    ("Wet", (20.0, 0.0, 50.0, 0.0)),
]
# hậu tố -> sun_altitude_angle (độ). Âm = mặt trời dưới đường chân trời = đêm.
_TIME = [("Noon", 75.0), ("Sunset", 15.0), ("Night", -90.0)]

COLS = ["w_cloud", "w_rain", "w_wet", "w_fog", "w_sun_alt"]


def from_preset(token: str):
    """Token preset CARLA -> dict 5 trường. Không nhận ra thì trả ClearNoon."""
    sky = next((v for k, v in _SKY if k.lower() in token.lower()), _SKY[-2][1])
    alt = next((v for k, v in _TIME if k.lower() in token.lower()), 75.0)
    cloud, rain, wet, fog = sky
    return dict(zip(COLS, [cloud, rain, wet, fog, alt]))


def from_carla_dict(w: dict):
    """`metadata.weather` của Practice/Redacted -> cùng 5 trường."""
    w = w or {}
    return {
        "w_cloud": float(w.get("cloudiness", 0.0)),
        "w_rain": float(w.get("precipitation", 0.0)),
        # wetness và precipitation_deposits nói cùng một chuyện (mặt đường ướt);
        # lấy cái lớn hơn để không bỏ sót trip chỉ set một trong hai.
        "w_wet": max(float(w.get("wetness", 0.0)),
                     float(w.get("precipitation_deposits", 0.0))),
        "w_fog": float(w.get("fog_density", 0.0)),
        "w_sun_alt": float(w.get("sun_altitude_angle", 75.0)),
    }


def empty():
    return {c: "" for c in COLS}
