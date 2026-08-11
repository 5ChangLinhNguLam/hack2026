output "review" {
  value = {
    region                    = var.aws_region
    instance_type             = var.instance_type
    network_mode              = "host"
    desired_min_max           = "1/1/1 (single-active, not HA)"
    road_channel              = local.road_channel
    cabin_channel             = local.cabin_channel
    decision_topic_filter     = local.decision_topic
    direct_ice_sender_cidrs   = var.direct_ice_sender_cidrs
    source_certificate_bound  = var.source_certificate_arn != ""
    android_certificate_bound = var.android_certificate_arn != ""
  }
}
