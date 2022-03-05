output "terraform_cloud_team_token" {
  value     = module.cloud.team_token
  sensitive = true
}
