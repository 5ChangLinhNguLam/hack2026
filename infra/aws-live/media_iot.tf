resource "awscc_kinesisvideo_signaling_channel" "road" {
  name                = local.road_channel
  type                = "SINGLE_MASTER"
  message_ttl_seconds = 60
}

resource "awscc_kinesisvideo_signaling_channel" "cabin" {
  name                = local.cabin_channel
  type                = "SINGLE_MASTER"
  message_ttl_seconds = 60
}

resource "aws_iot_thing" "source" {
  name = "${local.prefix}-source"
}

resource "aws_iot_thing" "android" {
  name = local.android_client_id
}

resource "aws_iot_policy" "source_credentials" {
  name = "${local.prefix}-source-credentials"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "iot:AssumeRoleWithCertificate"
      Resource = aws_iot_role_alias.source.arn
    }]
  })
}

resource "aws_iot_policy" "android_decisions" {
  name = "${local.prefix}-android-decisions"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "iot:Connect"
        Resource = "arn:${data.aws_partition.current.partition}:iot:${var.aws_region}:${data.aws_caller_identity.current.account_id}:client/${local.android_client_id}"
      },
      {
        Effect   = "Allow"
        Action   = "iot:Subscribe"
        Resource = "arn:${data.aws_partition.current.partition}:iot:${var.aws_region}:${data.aws_caller_identity.current.account_id}:topicfilter/${local.decision_topic}"
      },
      {
        Effect   = "Allow"
        Action   = "iot:Receive"
        Resource = "arn:${data.aws_partition.current.partition}:iot:${var.aws_region}:${data.aws_caller_identity.current.account_id}:topic/${local.publish_topic}"
      }
    ]
  })
}

resource "aws_iot_thing_principal_attachment" "source" {
  count = var.source_certificate_arn == "" ? 0 : 1

  thing     = aws_iot_thing.source.name
  principal = var.source_certificate_arn
}

resource "aws_iot_policy_attachment" "source" {
  count = var.source_certificate_arn == "" ? 0 : 1

  policy = aws_iot_policy.source_credentials.name
  target = var.source_certificate_arn
}

resource "aws_iot_thing_principal_attachment" "android" {
  count = var.android_certificate_arn == "" ? 0 : 1

  thing     = aws_iot_thing.android.name
  principal = var.android_certificate_arn
}

resource "aws_iot_policy_attachment" "android" {
  count = var.android_certificate_arn == "" ? 0 : 1

  policy = aws_iot_policy.android_decisions.name
  target = var.android_certificate_arn
}
