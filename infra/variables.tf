variable "region" { default = "us-east-1" }
variable "vpc_name" {}
variable "instance_name" {}
variable "instance_type" { default = "t3.micro" }
variable "bucket_name" {}
variable "user_data" { type = string }
variable "ssh_key_name" {
  description = "Existing AWS EC2 key pair name"
}
