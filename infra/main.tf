terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = "us-east-1"
  profile = "sam-dev"
}


module "vpc" {
  source   = "./modules/vpc"
  vpc_name = var.vpc_name
}

module "ec2" {
  source         = "./modules/ec2"
  instance_name  = var.instance_name
  instance_type  = var.instance_type
  subnet_id      = module.vpc.public_subnet_id
  vpc_id         = module.vpc.vpc_id
  user_data      = var.user_data
  ssh_key_name  = var.ssh_key_name
}

module "s3_static" {
  source      = "./modules/s3-static"
  bucket_name = var.bucket_name
}

