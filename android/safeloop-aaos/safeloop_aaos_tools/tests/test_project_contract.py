from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "app/src/main/AndroidManifest.xml"
JAVA_ROOT = ROOT / "app/src/main/java"
ANDROID = "{http://schemas.android.com/apk/res/android}"


def test_manifest_is_minimal_automotive_read_only_app() -> None:
    root = ET.parse(MANIFEST).getroot()
    permissions = {
        element.attrib[ANDROID + "name"]
        for element in root.findall("uses-permission")
    }
    features = {
        element.attrib[ANDROID + "name"]: element.attrib.get(ANDROID + "required")
        for element in root.findall("uses-feature")
    }
    application = root.find("application")
    assert application is not None
    activity = application.find("activity")
    assert activity is not None
    categories = {
        element.attrib[ANDROID + "name"]
        for element in activity.findall("intent-filter/category")
    }

    assert permissions == {
        "android.permission.INTERNET",
        "android.permission.ACCESS_NETWORK_STATE",
    }
    assert features["android.hardware.type.automotive"] == "true"
    assert "android.intent.category.CAR_LAUNCHER" in categories
    assert ANDROID + "appCategory" not in application.attrib
    assert not activity.findall("meta-data")


def test_runtime_has_no_webview_androidx_compose_or_vehicle_write_api() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in JAVA_ROOT.rglob("*.java"))
    forbidden = (
        "android.webkit",
        "androidx.",
        "android.car.hardware.property",
        "CarPropertyManager",
        "new WebView",
        "loadUrl(",
    )
    for token in forbidden:
        assert token not in source


def test_package_and_udp_port_are_stable() -> None:
    build = (ROOT / "app/build.gradle").read_text(encoding="utf-8")
    receiver = (
        JAVA_ROOT / "com/fptautomotive/safeloop/UdpDecisionReceiver.java"
    ).read_text(encoding="utf-8")

    assert 'applicationId "com.fptautomotive.safeloop"' in build
    assert "DEFAULT_PORT = 48100" in receiver
