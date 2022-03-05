terraform {
  required_providers {
    tfe = {
      version = "~> 0.26.0"
    }
  }
}

resource "tfe_organization" "this" {
  name  = var.organization
  email = var.email
}

resource "tfe_workspace" "this" {
  name         = var.workspace
  organization = tfe_organization.this.id
}

data "tfe_team" "this" {
  name         = "owners"
  organization = tfe_organization.this.id
}

resource "tfe_team_token" "this" {
  team_id = data.tfe_team.this.id
}

resource "tfe_variable" "aws_access_key_id" {
  key          = "aws_access_key_id"
  value        = var.aws_access_key_id
  category     = "terraform"
  sensitive    = false
  workspace_id = tfe_workspace.this.id
}

resource "tfe_variable" "aws_secret_access_key" {
  key          = "aws_secret_access_key"
  value        = var.aws_secret_access_key
  category     = "terraform"
  sensitive    = true
  workspace_id = tfe_workspace.this.id
}
