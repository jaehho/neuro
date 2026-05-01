SHELL := /bin/bash

.SILENT:
.DEFAULT_GOAL := help

.PHONY: help push-remote pull-remote push-remote-dry pull-remote-dry

## General
help: ## Show this help message
	echo "Available targets:"
	echo "=================="
	grep -hE '(^[a-zA-Z_%-]+:.*?## .*$$|^## )' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; \
		     /^## / {gsub("^## ", ""); print "\n\033[1;35m" $$0 "\033[0m"}; \
		     /^[a-zA-Z_%-]+:/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## Remote sync
# Bidirectional sync with the mililab mount at ~/mililab/neuro.
# Both directions use --update so newer files on the receiver are preserved; no --delete.
REMOTE_MOUNT := $(HOME)/mililab/home/jaeho/neuro

RSYNC_EXCLUDES := \
	--exclude=.git/ \
	--exclude=.venv/ \
	--exclude=__pycache__/ \
	--exclude=__marimo__/ \
	--exclude='*.pyc' \
	--exclude=.env \
	--exclude=.envrc \
	--exclude=data/ \
	--exclude=output/runs.db

RSYNC_FLAGS := -av --update

push-remote: ## Sync local -> mililab (code, scripts, notebooks)
	rsync $(RSYNC_FLAGS) $(RSYNC_EXCLUDES) ./ $(REMOTE_MOUNT)/

pull-remote: ## Sync mililab -> local (sweep outputs, remote-side edits)
	rsync $(RSYNC_FLAGS) $(RSYNC_EXCLUDES) $(REMOTE_MOUNT)/ ./

push-remote-dry: ## Dry-run of push-remote
	rsync $(RSYNC_FLAGS) --dry-run $(RSYNC_EXCLUDES) ./ $(REMOTE_MOUNT)/

pull-remote-dry: ## Dry-run of pull-remote
	rsync $(RSYNC_FLAGS) --dry-run $(RSYNC_EXCLUDES) $(REMOTE_MOUNT)/ ./
