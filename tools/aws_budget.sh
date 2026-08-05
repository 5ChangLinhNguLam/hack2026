#!/usr/bin/env bash
# aws_budget.sh — Xem ngân sách tháng hiện tại và chi tiêu tháng trước.
# Dùng: bash tools/aws_budget.sh
# Ghi chú:
#   - Budgets là service GLOBAL -> BẮT BUỘC --region us-east-1.
#   - Budget MONTHLY lưu lịch sử tháng hiện tại + tối đa 12 tháng trước.
#   - Dữ liệu billing của AWS trễ ~8-24h, không real-time.
set -uo pipefail

command -v aws >/dev/null || { echo ">> Chưa cài AWS CLI."; exit 1; }
command -v python3 >/dev/null || { echo ">> Chưa cài python3."; exit 1; }

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [ -z "$ACCOUNT_ID" ] || [ "$ACCOUNT_ID" = "None" ]; then
  echo ">> Chưa cấu hình AWS CLI (aws configure)."; exit 1;
fi

python3 - "$ACCOUNT_ID" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

ACCOUNT_ID = sys.argv[1]
REGION = "us-east-1"


class AwsError(RuntimeError):
    pass


def aws_json(*args):
    command = ["aws", *args, "--region", REGION, "--output", "json"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or "AWS CLI lỗi không rõ nguyên nhân"
        raise AwsError(message)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AwsError(f"AWS trả về JSON không hợp lệ: {exc}") from exc


def amount(value):
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(0)


def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def month_label(entry):
    start = parse_time(entry.get("TimePeriod", {}).get("Start"))
    return start.astimezone(timezone.utc).strftime("%Y-%m") if start else "?"


def latest_completed_period(entries):
    now = datetime.now(timezone.utc)
    completed = []
    for entry in entries:
        end = parse_time(entry.get("TimePeriod", {}).get("End"))
        if end and end.astimezone(timezone.utc) <= now:
            completed.append((end, entry))
    return max(completed, key=lambda item: item[0])[1] if completed else None


def print_threshold(pct):
    if pct >= 90:
        print("  🔴 >90% — dừng GPU ngay.")
    elif pct >= 80:
        print("  🟠 >80% — phanh gấp.")
    elif pct >= 50:
        print("  🟡 >50% — ưu tiên spot + tắt máy khi rảnh.")


try:
    budgets = aws_json(
        "budgets", "describe-budgets", "--account-id", ACCOUNT_ID
    ).get("Budgets", [])
except AwsError as exc:
    print(">> Không lấy được budget. AWS trả về:", file=sys.stderr)
    print(str(exc), file=sys.stderr)
    sys.exit(1)

print(f"Account: {ACCOUNT_ID}")
if not budgets:
    print(">> Account chưa có budget nào.")
    sys.exit(0)

for budget in budgets:
    name = budget.get("BudgetName", "?")
    time_unit = budget.get("TimeUnit", "")
    limit_data = budget.get("BudgetLimit", {})
    limit_value = amount(limit_data.get("Amount"))
    unit = limit_data.get("Unit", "")
    calculated = budget.get("CalculatedSpend", {})
    actual_value = amount(calculated.get("ActualSpend", {}).get("Amount"))
    forecast_raw = calculated.get("ForecastedSpend", {}).get("Amount")
    pct = actual_value / limit_value * 100 if limit_value else Decimal(0)

    print(f"\n=== {name}  ({time_unit}) ===")
    print("  THÁNG HIỆN TẠI")
    print(f"  Trần (limit)     : {limit_value:>12,.2f} {unit}")
    print(f"  Đã tiêu (actual) : {actual_value:>12,.2f} {unit}   ({pct:.1f}%)")
    print(f"  CÒN LẠI          : {limit_value - actual_value:>12,.2f} {unit}")
    if forecast_raw is not None:
        forecast = amount(forecast_raw)
        forecast_pct = forecast / limit_value * 100 if limit_value else Decimal(0)
        print(f"  Dự báo cuối kỳ   : {forecast:>12,.2f} {unit}   ({forecast_pct:.1f}%)")
    print_threshold(pct)

    if time_unit != "MONTHLY":
        print("\n  THÁNG TRƯỚC      : chỉ hỗ trợ budget MONTHLY")
        continue

    try:
        history = aws_json(
            "budgets", "describe-budget-performance-history",
            "--account-id", ACCOUNT_ID,
            "--budget-name", name,
        ).get("BudgetPerformanceHistory", {})
    except AwsError as exc:
        first_line = str(exc).splitlines()[0]
        print(f"\n  THÁNG TRƯỚC      : không lấy được ({first_line})")
        continue

    previous = latest_completed_period(history.get("BudgetedAndActualAmountsList", []))
    if previous is None:
        print("\n  THÁNG TRƯỚC      : chưa có dữ liệu")
        continue

    previous_budget = previous.get("BudgetedAmount", {})
    previous_actual = previous.get("ActualAmount", {})
    previous_limit = amount(previous_budget.get("Amount"))
    previous_spend = amount(previous_actual.get("Amount"))
    previous_unit = previous_actual.get("Unit") or previous_budget.get("Unit") or unit
    previous_pct = previous_spend / previous_limit * 100 if previous_limit else Decimal(0)

    print(f"\n  THÁNG TRƯỚC ({month_label(previous)})")
    print(f"  Trần (limit)     : {previous_limit:>12,.2f} {previous_unit}")
    print(f"  Đã tiêu (actual) : {previous_spend:>12,.2f} {previous_unit}   ({previous_pct:.1f}%)")
    print(f"  CÒN LẠI          : {previous_limit - previous_spend:>12,.2f} {previous_unit}")
PY
