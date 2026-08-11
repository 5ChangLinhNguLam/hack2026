data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "iot_credentials_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["credentials.iot.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_iam_role" "instance" {
  name               = "${local.prefix}-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "instance_ecs" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "instance" {
  name = "${local.prefix}-instance"
  role = aws_iam_role.instance.name
}

resource "aws_iam_role" "execution" {
  name               = "${local.prefix}-execution"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
}

data "aws_iam_policy_document" "execution" {
  statement {
    sid       = "EcrAuthorization"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "PullExactRepository"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.worker.arn]
  }
  statement {
    sid       = "WriteExactLogGroup"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.worker.arn}:*"]
  }
}

resource "aws_iam_role_policy" "execution" {
  name   = "least-privilege"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution.json
}

resource "aws_iam_role" "task" {
  name               = "${local.prefix}-task"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
}

data "aws_iam_policy_document" "task" {
  statement {
    sid = "ViewExactSignalingChannels"
    actions = [
      "kinesisvideo:ConnectAsViewer",
      "kinesisvideo:DescribeSignalingChannel",
      "kinesisvideo:GetIceServerConfig",
      "kinesisvideo:GetSignalingChannelEndpoint",
    ]
    resources = [
      awscc_kinesisvideo_signaling_channel.road.arn,
      awscc_kinesisvideo_signaling_channel.cabin.arn,
    ]
  }
  statement {
    sid       = "ConnectFixedWorkerClient"
    actions   = ["iot:Connect"]
    resources = ["arn:${data.aws_partition.current.partition}:iot:${var.aws_region}:${data.aws_caller_identity.current.account_id}:client/${local.worker_client_id}"]
  }
  statement {
    sid       = "PublishOnlyVehicleDecisions"
    actions   = ["iot:Publish"]
    resources = ["arn:${data.aws_partition.current.partition}:iot:${var.aws_region}:${data.aws_caller_identity.current.account_id}:topic/${local.publish_topic}"]
  }
}

resource "aws_iam_role_policy" "task" {
  name   = "least-privilege"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

resource "aws_iam_role" "source" {
  name               = "${local.prefix}-source"
  assume_role_policy = data.aws_iam_policy_document.iot_credentials_assume.json
}

data "aws_iam_policy_document" "source" {
  statement {
    sid = "MasterExactSignalingChannels"
    actions = [
      "kinesisvideo:ConnectAsMaster",
      "kinesisvideo:DescribeSignalingChannel",
      "kinesisvideo:GetIceServerConfig",
      "kinesisvideo:GetSignalingChannelEndpoint",
    ]
    resources = [
      awscc_kinesisvideo_signaling_channel.road.arn,
      awscc_kinesisvideo_signaling_channel.cabin.arn,
    ]
  }
}

resource "aws_iam_role_policy" "source" {
  name   = "least-privilege"
  role   = aws_iam_role.source.id
  policy = data.aws_iam_policy_document.source.json
}

resource "aws_iot_role_alias" "source" {
  alias               = "${local.prefix}-source"
  role_arn            = aws_iam_role.source.arn
  credential_duration = 3600
}
