variable "project" {
  type = string
  validation {
    condition     = can(regex("^[[:alpha:]]\\w*$", var.project))
    error_message = "A project name must be \"[A-Za-z][0-9A-Za-z_]*\"."
  }
}
variable "terraform_cloud_user_token" {
  type        = string
  sensitive   = true
  description = "Terraform Cloud user token"
}
variable "organization" {
  type        = string
  description = "Terraform Cloud organization ([A-Za-z][0-9A-Za-z_]*)"
}
variable "email" {
  type        = string
  description = "Terraform Cloud email"
}
variable "workspace" {
  type        = string
  description = "Terraform Cloud workspace ([A-Za-z][0-9A-Za-z_]*)"
}
variable "initialize_aws" {
  type    = bool
  default = false
}
variable "aws_access_key_id" {
  type        = string
  description = "administrative AWS access key ID"
}
variable "aws_secret_access_key" {
  type        = string
  sensitive   = true
  description = "administrative AWS secret access key"
}
