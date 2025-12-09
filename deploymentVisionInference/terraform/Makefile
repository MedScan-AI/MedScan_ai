# Makefile for Vision Inference API Terraform deployment

.PHONY: help init plan apply destroy output clean validate format

help:
	@echo "Vision Inference API - Terraform Commands"
	@echo ""
	@echo "Usage:"
	@echo "  make init       - Initialize Terraform"
	@echo "  make validate   - Validate Terraform configuration"
	@echo "  make format     - Format Terraform files"
	@echo "  make plan       - Show execution plan"
	@echo "  make apply      - Apply changes"
	@echo "  make destroy    - Destroy all resources"
	@echo "  make output     - Show all outputs"
	@echo "  make clean      - Clean Terraform files"
	@echo ""

init:
	@echo "Initializing Terraform..."
	terraform init

validate:
	@echo "Validating Terraform configuration..."
	terraform validate

format:
	@echo "Formatting Terraform files..."
	terraform fmt -recursive

plan:
	@echo "Creating execution plan..."
	terraform plan

apply:
	@echo "Applying Terraform configuration..."
	terraform apply

destroy:
	@echo "Destroying Terraform resources..."
	terraform destroy

output:
	@echo "=== Terraform Outputs ==="
	@terraform output -json | jq -r 'to_entries[] | "\(.key): \(.value.value)"'

clean:
	@echo "Cleaning Terraform files..."
	rm -rf .terraform
	rm -f .terraform.lock.hcl
	rm -f terraform.tfstate*
