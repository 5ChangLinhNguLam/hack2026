resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = local.prefix }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
  tags   = { Name = local.prefix }
}

resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.this.id
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  map_public_ip_on_launch = true

  tags = { Name = "${local.prefix}-public-${count.index + 1}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = { Name = "${local.prefix}-public" }
}

resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "worker" {
  name_prefix = "${local.prefix}-"
  description = "No SSH/TCP ingress; optional scoped UDP ICE ingress"
  vpc_id      = aws_vpc.this.id

  dynamic "ingress" {
    for_each = toset(var.direct_ice_sender_cidrs)
    content {
      description = "Direct WebRTC ICE from approved sender"
      protocol    = "udp"
      from_port   = 1024
      to_port     = 65535
      cidr_blocks = [ingress.value]
    }
  }

  dynamic "egress" {
    for_each = toset(var.direct_ice_sender_cidrs)
    content {
      description = "Direct WebRTC ICE to approved sender"
      protocol    = "udp"
      from_port   = 1024
      to_port     = 65535
      cidr_blocks = [egress.value]
    }
  }

  egress {
    description = "HTTPS, WSS, STUN and TURN over TCP 443"
    protocol    = "tcp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "STUN and TURN over UDP 443"
    protocol    = "udp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "DNS to VPC resolver"
    protocol    = "udp"
    from_port   = 53
    to_port     = 53
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    description = "DNS fallback over TCP"
    protocol    = "tcp"
    from_port   = 53
    to_port     = 53
    cidr_blocks = [var.vpc_cidr]
  }

  tags = { Name = local.prefix }
}
