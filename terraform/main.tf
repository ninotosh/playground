provider "aws" {
  region     = "us-east-1" # does not matter
  access_key = var.aws_access_key_id
  secret_key = var.aws_secret_access_key
  default_tags {
    tags = {
      project = var.project
    }
  }
}

module "aws" {
  source = "./modules/aws"
  count  = var.initialize_aws ? 1 : 0

  user_name = var.project
}

provider "tfe" {
  token = var.terraform_cloud_user_token
}

module "cloud" {
  source = "./modules/cloud"

  organization          = var.organization
  email                 = var.email
  workspace             = var.workspace
  aws_access_key_id     = length(module.aws) > 0 ? module.aws[0].access_key_id : null
  aws_secret_access_key = length(module.aws) > 0 ? module.aws[0].secret_access_key : null
}
