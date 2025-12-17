############################################
# Global
############################################
region = "us-east-1"

############################################
# VPC
############################################
vpc_name = "demo-vpc"

############################################
# EC2
############################################
instance_name = "demo-ec2"
instance_type = "t3.micro"

ssh_key_name = "mcp-bot-key"
user_data = <<EOF
#!/bin/bash
yum update -y



# Base tools
yum install -y git docker python3 python3-pip

# Docker
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# Python tooling
pip3 install --upgrade pip
pip3 install poetry boto3 requests
EOF

############################################
# S3 Static Website
############################################
bucket_name = "demo-static-site-unique-123456"
