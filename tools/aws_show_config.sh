#!/usr/bin/env bash
# aws_show_config.sh — Xem cấu hình tài nguyên trong AWS account:
#   EC2 instances (CPU / RAM / GPU) + EBS volumes (SSD) trên MỌI region.
#
# Yêu cầu: đã `aws configure` xong (xem hướng dẫn bên dưới).
# Dùng:    bash tools/aws_show_config.sh
#
# Diễn giải cột:
#   MemoryInfo.SizeInMiB      -> RAM tính bằng MiB (chia 1024 = GiB)
#   GpuInfo.*                 -> tên/số lượng/VRAM GPU (rỗng = KHÔNG có GPU)
#   InstanceStorageInfo       -> SSD NVMe gắn kèm (ephemeral, mất khi stop)
#   describe-volumes Size     -> dung lượng EBS (SSD gp3/gp2/io2...) tính bằng GiB
set -uo pipefail
AWS=aws

echo "==================== DANH TÍNH ===================="
$AWS sts get-caller-identity --output table || {
  echo ">> Chưa cấu hình credentials. Chạy: aws configure"; exit 1; }

echo
echo "==================== QUÉT EC2 MỌI REGION ===================="
regions=$($AWS ec2 describe-regions --all-regions \
          --query "Regions[].RegionName" --output text 2>/dev/null)
[ -z "$regions" ] && { echo ">> Không lấy được danh sách region."; exit 1; }

found=0
for r in $regions; do
  ids=$($AWS ec2 describe-instances --region "$r" \
        --query "Reservations[].Instances[].InstanceId" --output text 2>/dev/null)
  [ -z "$ids" ] && continue
  found=1

  echo
  echo "############### REGION: $r ###############"
  echo "--- Instances ---"
  $AWS ec2 describe-instances --region "$r" \
    --query "Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Placement.AvailabilityZone,PrivateIpAddress,PublicIpAddress]" \
    --output table

  types=$($AWS ec2 describe-instances --region "$r" \
    --query "Reservations[].Instances[].InstanceType" --output text \
    | tr '\t' '\n' | sort -u)

  echo "--- Cấu hình theo instance type (vCPU / RAM MiB / GPU / VRAM MiB / NVMe GB) ---"
  # shellcheck disable=SC2086
  $AWS ec2 describe-instance-types --region "$r" --instance-types $types \
    --query "InstanceTypes[].[InstanceType,VCpuInfo.DefaultVCpus,MemoryInfo.SizeInMiB,to_string(GpuInfo.Gpus[].Name),to_string(GpuInfo.Gpus[].Count),GpuInfo.TotalGpuMemoryInMiB,to_string(InstanceStorageInfo.TotalSizeInGB)]" \
    --output table

  echo "--- EBS volumes / SSD (VolId / Size GiB / Type / IOPS / State / GắnVào) ---"
  $AWS ec2 describe-volumes --region "$r" \
    --query "Volumes[].[VolumeId,Size,VolumeType,Iops,State,Attachments[0].InstanceId]" \
    --output table
done

[ "$found" -eq 0 ] && echo ">> Không có EC2 instance nào (account có thể chưa được tạo máy)."
echo
echo "==================== XONG ===================="
