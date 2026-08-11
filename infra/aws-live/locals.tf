data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ssm_parameter" "ecs_gpu_ami" {
  name = "/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id"
}

locals {
  prefix            = "${var.project}-${var.vehicle_id}"
  road_channel      = "${local.prefix}-road"
  cabin_channel     = "${local.prefix}-cabin"
  worker_client_id  = "${local.prefix}-worker"
  android_client_id = "${local.prefix}-android"
  decision_topic    = "safeloop/v2/vehicles/${var.vehicle_id}/sessions/+/decision"
  publish_topic     = "safeloop/v2/vehicles/${var.vehicle_id}/sessions/*/decision"

  common_tags = {
    Project      = var.project
    Vehicle      = var.vehicle_id
    Environment  = "poc"
    ManagedBy    = "terraform"
    DataBoundary = "truth-free-live-only"
  }
}
