#!/usr/bin/env bash
# aws_gpu_ctl.sh — Điều khiển GPU box dùng chung (Hướng A: 1 máy, 5 member chia lượt).
# Tự tìm máy theo tag Name=hackathon-gpu, khỏi nhớ InstanceId.
#
#   bash tools/aws_gpu_ctl.sh status              # trạng thái + IP + GPU đang bận?
#   bash tools/aws_gpu_ctl.sh start               # bật máy
#   bash tools/aws_gpu_ctl.sh stop                # tắt máy (tiết kiệm tiền)
#   bash tools/aws_gpu_ctl.sh ssh                 # SSH vào (dùng key của bạn)
#   bash tools/aws_gpu_ctl.sh gpu                 # xem nvidia-smi từ xa
#   bash tools/aws_gpu_ctl.sh add-key <file.pub>  # thêm 1 member (admin chạy)
#   bash tools/aws_gpu_ctl.sh list-keys           # xem ai đã được add
#   bash tools/aws_gpu_ctl.sh open-ip [IP]        # mở firewall SSH cho 1 IP (mặc định IP của bạn)
#   bash tools/aws_gpu_ctl.sh eip                 # gán IP tĩnh (khỏi đổi IP mỗi lần start)
#
# Key SSH: ưu tiên arg > biến môi trường KEY > mặc định ~/.ssh/hackathon-gpu.pem
#   Member dùng key riêng:  KEY=~/.ssh/hackathon bash tools/aws_gpu_ctl.sh ssh
set -uo pipefail

REGION=ap-southeast-1
TAG_NAME=hackathon-gpu
SSH_USER=ubuntu
DEFAULT_KEY="$HOME/.ssh/hackathon-gpu.pem"

die(){ printf '>> LỖI: %s\n' "$*" >&2; exit 1; }
is_empty(){ [ -z "${1:-}" ] || [ "$1" = "None" ]; }
command -v aws >/dev/null || die "chưa có aws CLI"

get_iid(){
  aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=$TAG_NAME" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query "Reservations[].Instances[].InstanceId" --output text 2>/dev/null | awk '{print $1}'
}
get_field(){ # $1=iid  $2=jmespath
  aws ec2 describe-instances --region "$REGION" --instance-ids "$1" \
    --query "Reservations[0].Instances[0].$2" --output text 2>/dev/null
}
get_sg(){ get_field "$1" "SecurityGroups[0].GroupId"; }
pick_key(){ echo "${1:-${KEY:-$DEFAULT_KEY}}"; }

IID=$(get_iid)
CMD="${1:-status}"

case "$CMD" in
  status)
    is_empty "$IID" && { echo "Không có máy nào (tag=$TAG_NAME). Tạo bằng: bash tools/aws_launch_gpu.sh --go"; exit 0; }
    STATE=$(get_field "$IID" "State.Name")
    IP=$(get_field "$IID" "PublicIpAddress")
    TYPE=$(get_field "$IID" "InstanceType")
    echo "InstanceId : $IID"
    echo "Type       : $TYPE"
    echo "State      : $STATE"
    echo "Public IP  : ${IP:-<không có (máy đang tắt)>}"
    if [ "$STATE" = "running" ] && ! is_empty "$IP"; then
      echo "SSH        : ssh -i <key> $SSH_USER@$IP"
    fi
    ;;

  start)
    is_empty "$IID" && die "chưa có máy. Tạo bằng: bash tools/aws_launch_gpu.sh --go"
    echo "Đang bật $IID..."
    aws ec2 start-instances --region "$REGION" --instance-ids "$IID" >/dev/null
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
    IP=$(get_field "$IID" "PublicIpAddress")
    echo "✅ Đang chạy. Public IP MỚI: $IP"
    echo "   (IP đổi sau mỗi lần start — chạy 'eip' 1 lần để cố định IP cho cả team.)"
    ;;

  stop)
    is_empty "$IID" && die "không tìm thấy máy."
    echo "Đang tắt $IID (dữ liệu trên ổ vẫn còn)..."
    aws ec2 stop-instances --region "$REGION" --instance-ids "$IID" >/dev/null
    echo "✅ Đã gửi lệnh stop. Lúc tắt chỉ còn tính tiền ổ đĩa (~\$20/tháng)."
    ;;

  ssh)
    is_empty "$IID" && die "không tìm thấy máy."
    IP=$(get_field "$IID" "PublicIpAddress"); is_empty "$IP" && die "máy chưa chạy (start trước)."
    KEYFILE=$(pick_key "${2:-}")
    [ -f "$KEYFILE" ] || die "không thấy key: $KEYFILE (member dùng: KEY=~/.ssh/hackathon bash ... ssh)"
    echo "SSH bằng key: $KEYFILE"
    exec ssh -i "$KEYFILE" -o StrictHostKeyChecking=accept-new "$SSH_USER@$IP"
    ;;

  gpu)
    is_empty "$IID" && die "không tìm thấy máy."
    IP=$(get_field "$IID" "PublicIpAddress"); is_empty "$IP" && die "máy chưa chạy."
    KEYFILE=$(pick_key "${2:-}")
    ssh -i "$KEYFILE" -o StrictHostKeyChecking=accept-new "$SSH_USER@$IP" nvidia-smi
    ;;

  add-key)
    is_empty "$IID" && die "không tìm thấy máy."
    PUB="${2:-}"; [ -f "$PUB" ] || die "cần file .pub: bash tools/aws_gpu_ctl.sh add-key member.pub"
    grep -q '^ssh-' "$PUB" || die "$PUB không giống public key (phải bắt đầu bằng ssh-...)."
    IP=$(get_field "$IID" "PublicIpAddress"); is_empty "$IP" && die "máy chưa chạy (start trước)."
    KEYFILE=$(pick_key "")   # dùng key admin để vào thêm khoá
    [ -f "$KEYFILE" ] || die "cần key admin $KEYFILE để thêm member."
    PUBLINE=$(cat "$PUB")
    ssh -i "$KEYFILE" -o StrictHostKeyChecking=accept-new "$SSH_USER@$IP" \
      "mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && \
       grep -qxF '$PUBLINE' ~/.ssh/authorized_keys || echo '$PUBLINE' >> ~/.ssh/authorized_keys; \
       chmod 600 ~/.ssh/authorized_keys" \
      && echo "✅ Đã thêm member. Họ SSH bằng: ssh -i ~/.ssh/hackathon $SSH_USER@$IP"
    ;;

  list-keys)
    is_empty "$IID" && die "không tìm thấy máy."
    IP=$(get_field "$IID" "PublicIpAddress"); is_empty "$IP" && die "máy chưa chạy."
    KEYFILE=$(pick_key "")
    ssh -i "$KEYFILE" -o StrictHostKeyChecking=accept-new "$SSH_USER@$IP" \
      "awk '{print NR\": \"\$NF}' ~/.ssh/authorized_keys"
    ;;

  open-ip)
    is_empty "$IID" && die "không tìm thấy máy."
    SG=$(get_sg "$IID"); is_empty "$SG" && die "không lấy được security group."
    NEWIP="${2:-}"
    is_empty "$NEWIP" && NEWIP=$(curl -s --max-time 10 https://checkip.amazonaws.com | tr -d '\n')
    is_empty "$NEWIP" && die "không xác định được IP."
    aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
      --protocol tcp --port 22 --cidr "${NEWIP}/32" >/dev/null 2>&1 \
      && echo "✅ Đã mở SSH cho ${NEWIP}/32" \
      || echo "(IP ${NEWIP}/32 có thể đã được mở trước đó — OK)"
    ;;

  eip)
    is_empty "$IID" && die "không tìm thấy máy."
    STATE=$(get_field "$IID" "State.Name")
    [ "$STATE" = "running" ] || die "bật máy trước (start) rồi mới gán IP tĩnh."
    ALLOC=$(aws ec2 allocate-address --region "$REGION" --domain vpc --query AllocationId --output text)
    aws ec2 associate-address --region "$REGION" --instance-id "$IID" --allocation-id "$ALLOC" >/dev/null
    IP=$(get_field "$IID" "PublicIpAddress")
    echo "✅ Đã gán IP tĩnh: $IP (không đổi nữa dù stop/start)."
    echo "   Lưu ý: EIP tính phí nhẹ (~\$3.6/tháng) khi máy TẮT. Không cần nữa thì release."
    ;;

  *) grep '^#' "$0" | sed 's/^# \{0,1\}//' ;;   # in help từ header
esac
