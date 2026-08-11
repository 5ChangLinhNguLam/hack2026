terraform {
  required_version = ">= 1.7.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.80.0, < 7.0.0"
    }
    awscc = {
      source  = "hashicorp/awscc"
      version = ">= 1.0.0, < 2.0.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.common_tags
  }
}

provider "awscc" {
  region = var.aws_region
}
