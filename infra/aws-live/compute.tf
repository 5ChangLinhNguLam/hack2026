resource "aws_ecr_repository" "worker" {
  name                 = "${var.project}/live-worker"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false

  encryption_configuration {
    encryption_type = "AES256"
  }

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "worker" {
  repository = aws_ecr_repository.worker.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the ten newest immutable images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/ecs/${local.prefix}/live-worker"
  retention_in_days = 14
}

resource "aws_ecs_cluster" "this" {
  name = local.prefix

  setting {
    name  = "containerInsights"
    value = "disabled"
  }
}

resource "aws_launch_template" "worker" {
  name_prefix   = "${local.prefix}-"
  image_id      = data.aws_ssm_parameter.ecs_gpu_ami.value
  instance_type = var.instance_type

  iam_instance_profile {
    arn = aws_iam_instance_profile.instance.arn
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
    instance_metadata_tags      = "disabled"
  }

  network_interfaces {
    associate_public_ip_address = true
    delete_on_termination       = true
    device_index                = 0
    security_groups             = [aws_security_group.worker.id]
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      encrypted             = true
      delete_on_termination = true
      volume_type           = "gp3"
      volume_size           = var.root_volume_gib
    }
  }

  user_data = base64encode(<<-EOT
    #!/bin/bash
    echo 'ECS_CLUSTER=${aws_ecs_cluster.this.name}' >> /etc/ecs/ecs.config
    echo 'ECS_ENABLE_GPU_SUPPORT=true' >> /etc/ecs/ecs.config
    echo 'ECS_ENABLE_TASK_IAM_ROLE_NETWORK_HOST=true' >> /etc/ecs/ecs.config
  EOT
  )

  tag_specifications {
    resource_type = "instance"
    tags          = merge(local.common_tags, { Name = local.prefix })
  }

  lifecycle {
    create_before_destroy = true

    precondition {
      condition = (
        (var.aws_region == "ap-southeast-1" && var.instance_type == "g4dn.xlarge") ||
        (var.aws_region == "ap-southeast-5" && var.instance_type == "g6.xlarge")
      )
      error_message = "Use Singapore g4dn.xlarge, or move the whole fallback stack to Malaysia g6.xlarge."
    }
  }
}

resource "aws_autoscaling_group" "worker" {
  name_prefix         = "${local.prefix}-"
  min_size            = 1
  desired_capacity    = 1
  max_size            = 1
  vpc_zone_identifier = aws_subnet.public[*].id
  health_check_type   = "EC2"

  launch_template {
    id      = aws_launch_template.worker.id
    version = "$Latest"
  }

  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 0
    }
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = "true"
    propagate_at_launch = true
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_ecs_capacity_provider" "worker" {
  name = local.prefix

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.worker.arn
    managed_termination_protection = "DISABLED"

    managed_scaling {
      status                    = "DISABLED"
      target_capacity           = 100
      minimum_scaling_step_size = 1
      maximum_scaling_step_size = 1
    }
  }
}

resource "aws_ecs_cluster_capacity_providers" "worker" {
  cluster_name       = aws_ecs_cluster.this.name
  capacity_providers = [aws_ecs_capacity_provider.worker.name]

  default_capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.worker.name
    weight            = 1
    base              = 1
  }
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.prefix}-live-worker"
  network_mode             = "host"
  requires_compatibilities = ["EC2"]
  cpu                      = "3584"
  memory                   = "14336"
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name                   = "live-worker"
    image                  = var.container_image
    essential              = true
    readonlyRootFilesystem = true
    user                   = "65532:65532"
    environment = [
      { name = "AWS_REGION", value = var.aws_region },
      { name = "SAFELOOP_VEHICLE_ID", value = var.vehicle_id },
      { name = "SAFELOOP_ROAD_CHANNEL", value = awscc_kinesisvideo_signaling_channel.road.name },
      { name = "SAFELOOP_CABIN_CHANNEL", value = awscc_kinesisvideo_signaling_channel.cabin.name },
      { name = "SAFELOOP_IOT_CLIENT_ID", value = local.worker_client_id },
      { name = "SAFELOOP_DECISION_TOPIC_TEMPLATE", value = "safeloop/v2/vehicles/${var.vehicle_id}/sessions/{session_id}/decision" },
    ]
    resourceRequirements = [{ type = "GPU", value = "1" }]
    linuxParameters = {
      initProcessEnabled = true
      capabilities       = { drop = ["ALL"] }
      tmpfs = [{
        containerPath = "/tmp"
        size          = 1024
        mountOptions  = ["rw", "nosuid", "nodev", "noexec"]
      }]
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.worker.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "worker"
      }
    }
  }])

  runtime_platform {
    cpu_architecture        = "X86_64"
    operating_system_family = "LINUX"
  }
}

resource "aws_ecs_service" "worker" {
  name            = "live-worker"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = 1

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.worker.name
    weight            = 1
    base              = 1
  }

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100
  enable_execute_command             = false
  propagate_tags                     = "SERVICE"

  depends_on = [aws_ecs_cluster_capacity_providers.worker]
}

resource "aws_cloudwatch_metric_alarm" "instance_missing" {
  alarm_name          = "${local.prefix}-instance-missing"
  alarm_description   = "Single-active PoC has no in-service GPU host"
  namespace           = "AWS/AutoScaling"
  metric_name         = "GroupInServiceInstances"
  dimensions          = { AutoScalingGroupName = aws_autoscaling_group.worker.name }
  statistic           = "Minimum"
  period              = 60
  evaluation_periods  = 2
  comparison_operator = "LessThanThreshold"
  threshold           = 1
  treat_missing_data  = "breaching"
}

resource "aws_budgets_budget" "monthly" {
  count = var.budget_notification_email == "" ? 0 : 1

  name         = "${local.prefix}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = [format("user:Project$%s", var.project)]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_notification_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_notification_email]
  }
}
