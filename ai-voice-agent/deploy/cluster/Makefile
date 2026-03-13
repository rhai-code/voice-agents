# AI Voice Agent - Models (Kustomize) Makefile
# Run `make help` to see available targets.

.PHONY: help install install-help install-status uninstall

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
help: ## Show help for model deployment targets
	@echo "AI Voice Agent - Model Deployment Commands"
	@echo "=========================================="
	@echo ""
	@echo "☸️  Model Commands:"
	@grep -E '^(install|install-help|install-status|uninstall):.*?## ' $(MAKEFILE_LIST) | \
		sed -e 's/^/  /' -e 's/:.*## / - /'
	@echo ""
	@echo "For detailed deployment help, run: make install-help"

# -----------------------------------------------------------------------------
# Deployment (Kustomize/OpenShift)
# -----------------------------------------------------------------------------

# Kustomize deployment configuration variables
NAMESPACE ?=
MODELS_DIR := ../models

# Check namespace is set only for commands that need it
ifneq (,$(filter install install-status uninstall,$(MAKECMDGOALS)))
ifeq ($(NAMESPACE),)
$(error NAMESPACE is not set. Use: make <target> NAMESPACE=<your-namespace>)
endif
endif

install-help: ## Show detailed deployment help and configuration options
	@echo "AI Voice Agent Model Deployment Help"
	@echo "===================================="
	@echo ""
	@echo "Available model targets:"
	@echo "  install          - Apply the models kustomization"
	@echo "  install-status   - Check model deployment status"
	@echo "  uninstall        - Delete the models kustomization"
	@echo ""
	@echo "Required Configuration:"
	@echo "  NAMESPACE        - Target namespace (required for all commands)"
	@echo ""
	@echo "Example usage:"
	@echo "  make install NAMESPACE=my-namespace"
	@echo "  make install-status NAMESPACE=my-namespace"

install: ## Apply the models kustomization
	@echo "Applying models kustomization to namespace $(NAMESPACE)"
	@oc apply -k $(MODELS_DIR) -n $(NAMESPACE)
	@echo "Models applied successfully"

uninstall: ## Uninstall deployment and clean up resources
	@echo "Deleting models kustomization from namespace $(NAMESPACE)"
	@oc delete -k $(MODELS_DIR) -n $(NAMESPACE) --ignore-not-found
	@echo "Models removed successfully"

install-status: ## Check model deployment status
	@echo "Model deployment status for namespace: $(NAMESPACE)"
	@echo "=============================================="
	@echo ""
	@echo "LLMInferenceServices:"
	@oc get llminferenceservices -n $(NAMESPACE) || true
	@echo ""
	@echo "Deployments:"
	@oc get deployments -n $(NAMESPACE) || true
	@echo ""
	@echo "Services:"
	@oc get svc -n $(NAMESPACE) || true
