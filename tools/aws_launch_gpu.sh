#!/usr/bin/env bash
# aws_launch_gpu.sh — Tạo 1 GPU instance để fine-tune CV, kèm auto-stop 23:00.
#
#   XEM KẾ HOẠCH (không tạo máy, không tốn tiền):  bash tools/aws_launch_gpu.sh
#   TẠO MÁY THẬT:                                   bash tools/aws_launch_gpu.sh --go
#
# Sau khi tạo, quản lý bằng tools/aws_gpu_ctl.sh (start/stop/ssh/terminate).
set -uo pipefail

# ============ CẤU HÌNH (sửa ở đây nếu cần) ============
REGION=ap-southeast-1
INSTANCE_TYPE=g4dn.xlarge          # T4 16GB. Đổi 'g5.xlarge' nếu cần A10G 24GB.
VOLUME_SIZE=200                    # GB, ổ gp3 (dataset + checkpoint)
KEY_NAME=hackathon-gpu
SG_NAME=hackathon-gpu-sg
TAG_NAME=hackathon-gpu
TZ_LOCAL=Asia/Ho_Chi_Minh
STOP_CRON="0 23"                   # 'phút giờ' -> 23:00 giờ VN tự shutdown->stop
PRICE_HINT="~\$0.74/h (on-demand g4dn.xlarge)"
# ======================================================

GO=0; [ "${1:-}" = "--go" ] && GO=1
KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"

say(){ printf '%s\n' "$*"; }
die(){ printf '>> LỖI: %s\n' "$*" >&2; exit 1; }
is_empty(){ [ -z "${1:-}" ] || [ "$1" = "None" ]; }

command -v aws >/dev/null || die "chưa có aws CLI"
ACCID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) \
  || die "chưa cấu hình credentials (aws configure)"

say "============================================================"
say " $([ $GO = 1 ] && echo 'CHẾ ĐỘ: TẠO MÁY THẬT (--go)' || echo 'CHẾ ĐỘ: PLAN (chỉ xem, không tạo máy)')"
say " Account $ACCID | Region $REGION | Type $INSTANCE_TYPE | Disk ${VOLUME_SIZE}GB gp3"
say " Giá ước tính: $PRICE_HINT  — nhớ STOP khi không dùng!"
say "============================================================"

# 1) Deep Learning AMI (PyTorch, Ubuntu 22.04) mới nhất
say "-- [1/5] Tìm Deep Learning AMI..."
read -r AMI_ID AMI_NAME < <(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*" \
            "Name=state,Values=available" "Name=architecture,Values=x86_64" \
  --query "sort_by(Images,&CreationDate)[-1].[ImageId,Name]" --output text 2>/dev/null)
if is_empty "${AMI_ID:-}"; then   # fallback rộng hơn
  read -r AMI_ID AMI_NAME < <(aws ec2 describe-images --region "$REGION" --owners amazon \
    --filters "Name=name,Values=Deep Learning*Ubuntu 22.04*" "Name=state,Values=available" \
              "Name=architecture,Values=x86_64" \
    --query "sort_by(Images,&CreationDate)[-1].[ImageId,Name]" --output text 2>/dev/null)
fi
is_empty "${AMI_ID:-}" && die "không tìm được Deep Learning AMI — cần chỉ định thủ công."
ROOT_DEV=$(aws ec2 describe-images --image-ids "$AMI_ID" --region "$REGION" \
  --query "Images[0].RootDeviceName" --output text)
say "   AMI = $AMI_ID"
say "         $AMI_NAME  (root=$ROOT_DEV)"

# 2) VPC + subnet mặc định
say "-- [2/5] VPC/subnet mặc định..."
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text)
is_empty "$VPC_ID" && die "không có default VPC — cần chỉ định subnet thủ công."
SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
  --query "Subnets[0].SubnetId" --output text)
is_empty "$SUBNET_ID" && SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0].SubnetId" --output text)
say "   VPC=$VPC_ID  Subnet=$SUBNET_ID"

# 3) IP hiện tại (để mở SSH đúng mình bạn)
MYIP=$(curl -s --max-time 10 https://checkip.amazonaws.com | tr -d '\n')
is_empty "$MYIP" && die "không lấy được IP public của bạn."
say "-- [3/5] SSH sẽ chỉ mở cho IP của bạn: ${MYIP}/32"

# 4) Key pair
say "-- [4/5] Key pair '$KEY_NAME'..."
if aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
  say "   đã tồn tại (dùng lại). Cần file $KEY_PATH để SSH."
elif [ "$GO" = 1 ]; then
  mkdir -p "$HOME/.ssh"
  aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
    --query KeyMaterial --output text > "$KEY_PATH"
  chmod 400 "$KEY_PATH"
  say "   đã tạo mới -> $KEY_PATH (giữ kỹ, đây là chìa khoá SSH)"
else
  say "   (PLAN) sẽ tạo mới -> $KEY_PATH"
fi

# 5) Security group (chỉ SSH 22 từ IP của bạn)
say "-- [5/5] Security group '$SG_NAME'..."
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null)
if is_empty "$SG_ID"; then
  if [ "$GO" = 1 ]; then
    SG_ID=$(aws ec2 create-security-group --region "$REGION" --group-name "$SG_NAME" \
      --description "Hackathon GPU SSH only" --vpc-id "$VPC_ID" --query GroupId --output text)
    aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
      --protocol tcp --port 22 --cidr "${MYIP}/32" >/dev/null
    say "   đã tạo $SG_ID, mở SSH từ ${MYIP}/32"
  else
    say "   (PLAN) sẽ tạo mới, mở SSH(22) từ ${MYIP}/32"
  fi
else
  say "   đã tồn tại: $SG_ID"
  [ "$GO" = 1 ] && aws ec2 authorize-security-group-ingress --region "$REGION" \
    --group-id "$SG_ID" --protocol tcp --port 22 --cidr "${MYIP}/32" >/dev/null 2>&1 \
    && say "   (đã thêm SSH cho ${MYIP}/32)" || true
fi

# user-data: đặt timezone + cron auto-stop 23:00
UD=$(mktemp)
cat > "$UD" <<EOF
#!/bin/bash
timedatectl set-timezone $TZ_LOCAL
echo '$STOP_CRON * * * root /sbin/shutdown -h now' > /etc/cron.d/hackathon-autostop
chmod 644 /etc/cron.d/hackathon-autostop
systemctl restart cron 2>/dev/null || service cron restart 2>/dev/null || true
EOF
BDM="[{\"DeviceName\":\"$ROOT_DEV\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"

if [ "$GO" = 0 ]; then
  say ""
  say ">> ĐÂY LÀ PLAN. Chưa tạo gì cả, chưa tốn tiền."
  say ">> Ưng rồi thì chạy:  bash tools/aws_launch_gpu.sh --go"
  rm -f "$UD"; exit 0
fi

say ""
say "-- Đang tạo instance..."
IID=$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" --subnet-id "$SUBNET_ID" \
  --block-device-mappings "$BDM" \
  --instance-initiated-shutdown-behavior stop \
  --user-data "file://$UD" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG_NAME},{Key=Team,Value=5ChangLinhNguLam}]" \
  --query "Instances[0].InstanceId" --output text)
rm -f "$UD"
is_empty "$IID" && die "tạo instance thất bại."
say "   InstanceId = $IID  (đang khởi động...)"

aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

say ""
say "============================================================"
say " ✅ XONG. Máy đang chạy (tính tiền từ bây giờ!)."
say "   InstanceId : $IID"
say "   Public IP  : $IP"
say "   SSH        : ssh -i $KEY_PATH ubuntu@$IP"
say "   Auto-stop  : 23:00 giờ VN mỗi ngày (tự shutdown->stop)"
say ""
say "   TẮT MÁY khi nghỉ (chỉ còn tính tiền ổ đĩa ~\$20/tháng):"
say "     aws ec2 stop-instances  --region $REGION --instance-ids $IID"
say "   BẬT LẠI:"
say "     aws ec2 start-instances --region $REGION --instance-ids $IID"
say "   XOÁ HẲN (mất dữ liệu trên ổ):"
say "     aws ec2 terminate-instances --region $REGION --instance-ids $IID"
say "============================================================"
