TERRAFORM_IMAGE=hashicorp/terraform
YQ_IMAGE=mikefarah/yq

.PHONY: all
all: config init apply

_TERRAFORM=@docker container run -it --rm -v `pwd`:/mnt -w /mnt \
-e "TF_VAR_terraform_cloud_user_token=${TF_VAR_terraform_cloud_user_token}" \
-e "TF_VAR_aws_access_key_id=${TF_VAR_aws_access_key_id}" \
-e "TF_VAR_aws_secret_access_key=${TF_VAR_aws_secret_access_key}" \
${TERRAFORM_IMAGE}

.PHONY: config
config:
	$(eval count := $(shell cat ../../config.yml | docker container run -i --rm ${YQ_IMAGE} eval '.servers[] | [select(.provider == "aws")] | length' -))
	@echo "initialize_aws = `test ${count} -gt 0 && echo true || echo false`" \
	> generated.initialize_aws.auto.tfvars
	@cat ../../config.yml | \
	docker container run -i --rm ${YQ_IMAGE} \
	eval '{"project": .project} * .terraform_cloud' - -o json \
	> generated.config.auto.tfvars.json

.PHONY: init
init:
	${_TERRAFORM} init

.PHONY: fmt
fmt:
	${_TERRAFORM} fmt -recursive

.PHONY: plan
plan:
	${_TERRAFORM} plan

.PHONY: apply
apply:
	${_TERRAFORM} apply

.PHONY: output
output:
	${_TERRAFORM} output ${VALUE}

.PHONY: terraform_cloud_team_token
terraform_cloud_team_token:
	@echo `$(MAKE) VALUE="-raw terraform_cloud_team_token" output`

.PHONY: refresh
refresh:
	${_TERRAFORM} apply -refresh-only

.PHONY: delete-local
delete-local:
	$(RM) -r .terraform
	$(RM) .terraform.lock.hcl
	$(RM) terraform.tfstate terraform.tfstate.backup
	$(RM) terraform.tfvars
	$(RM) generated.*.json

.PHONY: _delete-remote
_delete-remote:
	${_TERRAFORM} destroy

.PHONY: delete-remote
delete-remote: init _delete-remote

.PHONY: delete-all
delete-all: delete-remote delete-local

.PHONY: remove-image
remove-image:
	docker image rm -f ${TERRAFORM_IMAGE}
	docker image rm -f ${YQ_IMAGE}
