variable "aws_region" {
  description = "Primary PoC region. Keep Singapore unless the T4 soak fails."
  type        = string
  default     = "ap-southeast-1"
}

variable "project" {
  type    = string
  default = "safeloop-carsky-poc"

  validation {
    condition     = can(regex("^[a-z0-9-]{3,40}$", var.project))
    error_message = "project must be a 3-40 character lowercase slug."
  }
}

variable "vehicle_id" {
  description = "One test vehicle/Android Thing name."
  type        = string
  default     = "vehicle-poc-01"

  validation {
    condition     = can(regex("^[A-Za-z0-9_-]{1,64}$", var.vehicle_id))
    error_message = "vehicle_id may contain only letters, digits, '_' and '-'."
  }
}

variable "instance_type" {
  description = "Singapore primary is g4dn.xlarge. Malaysia fallback is g6.xlarge."
  type        = string
  default     = "g4dn.xlarge"

  validation {
    condition     = contains(["g4dn.xlarge", "g6.xlarge"], var.instance_type)
    error_message = "Only the reviewed T4 primary or L4 fallback is allowed."
  }
}

variable "container_image" {
  description = "Immutable ECR image URI. A mutable tag is rejected."
  type        = string
  default     = "000000000000.dkr.ecr.ap-southeast-1.amazonaws.com/not-built@sha256:0000000000000000000000000000000000000000000000000000000000000000"

  validation {
    condition     = can(regex("^[^:@]+(?:[.:][^:@]+)*/[^:@]+@sha256:[0-9a-f]{64}$", var.container_image))
    error_message = "container_image must be an immutable repository@sha256 URI."
  }
}

variable "vpc_cidr" {
  type    = string
  default = "10.42.0.0/16"
}

variable "direct_ice_sender_cidrs" {
  description = "Known source CIDRs allowed to reach host UDP ICE ports. Empty forces TURN-compatible egress-only behavior."
  type        = list(string)
  default     = []

  validation {
    condition     = alltrue([for cidr in var.direct_ice_sender_cidrs : can(cidrnetmask(cidr))])
    error_message = "Every direct_ice_sender_cidrs entry must be a valid CIDR."
  }
}

variable "root_volume_gib" {
  type    = number
  default = 100

  validation {
    condition     = var.root_volume_gib >= 50 && var.root_volume_gib <= 200
    error_message = "root_volume_gib must be between 50 and 200 GiB."
  }
}

variable "source_certificate_arn" {
  description = "Optional externally provisioned source certificate ARN. Never generate private keys in Terraform state."
  type        = string
  default     = ""
}

variable "android_certificate_arn" {
  description = "Optional externally provisioned Android certificate ARN. Private key belongs in Android Keystore."
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  type    = number
  default = 750

  validation {
    condition     = var.monthly_budget_usd > 0
    error_message = "monthly_budget_usd must be positive."
  }
}

variable "budget_notification_email" {
  description = "If non-empty, create 80% forecast and 100% actual budget notifications."
  type        = string
  default     = ""
}
