variable "user_name" {
  type = string
  validation {
    condition     = can(regex("^[[:alpha:]]\\w*$", var.user_name))
    error_message = "A user name must be \"[[:alpha:]]\\w*\"."
  }
}
