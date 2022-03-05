variable "organization" {
  type = string
  validation {
    condition     = can(regex("^[[:alpha:]]\\w*$", var.organization))
    error_message = "An organization must be \"[A-Za-z][0-9A-Za-z_]*\"."
  }
}

variable "email" {
  type = string
}

variable "workspace" {
  type = string
  validation {
    condition     = can(regex("^[[:alpha:]]\\w*$", var.workspace))
    error_message = "A workspace must be \"[A-Za-z][0-9A-Za-z_]*\"."
  }
}

variable "aws_access_key_id" {
  type = string
}

variable "aws_secret_access_key" {
  type      = string
  sensitive = true
}
