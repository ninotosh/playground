output "organization_name" {
  value = tfe_organization.this.id
}

output "workspace_name" {
  value = tfe_workspace.this.name
}

output "team_token" {
  value     = tfe_team_token.this.token
  sensitive = true
}
