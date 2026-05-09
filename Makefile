SHELL := /bin/bash

.SILENT:
.DEFAULT_GOAL := help

.PHONY: help push-remote pull-remote push-remote-dry pull-remote-dry pull-remote-output sweep-remote remote-shell

## General
help: ## Show this help message
	echo "Available targets:"
	echo "=================="
	grep -hE '(^[a-zA-Z_%-]+:.*?## .*$$|^## )' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; \
		     /^## / {gsub("^## ", ""); print "\n\033[1;35m" $$0 "\033[0m"}; \
		     /^[a-zA-Z_%-]+:/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## Remote sync (mililab)
# SSH host 'mililab' is configured in ~/.ssh/config (proxied via ice).
# A separate SSHFS mount at ~/mililab is used for browsing files; the make
# targets below use SSH+rsync directly so they work regardless of the mount.
REMOTE_HOST := mililab
REMOTE_PROJ := /home/jaeho/neuro

RSYNC_EXCLUDES := \
	--exclude=.git/ \
	--exclude=.venv/ \
	--exclude=__pycache__/ \
	--exclude=__marimo__/ \
	--exclude='*.pyc' \
	--exclude=.env \
	--exclude=.envrc \
	--exclude=data/

RSYNC_FLAGS := -av --update

push-remote: ## Push code to mililab (excludes output/ and runs.db)
	rsync $(RSYNC_FLAGS) $(RSYNC_EXCLUDES) --exclude=output/ ./ $(REMOTE_HOST):$(REMOTE_PROJ)/

pull-remote: ## Pull code edits from mililab (excludes output/ and runs.db)
	rsync $(RSYNC_FLAGS) $(RSYNC_EXCLUDES) --exclude=output/ $(REMOTE_HOST):$(REMOTE_PROJ)/ ./

push-remote-dry: ## Dry-run of push-remote
	rsync $(RSYNC_FLAGS) --dry-run $(RSYNC_EXCLUDES) --exclude=output/ ./ $(REMOTE_HOST):$(REMOTE_PROJ)/

pull-remote-dry: ## Dry-run of pull-remote
	rsync $(RSYNC_FLAGS) --dry-run $(RSYNC_EXCLUDES) --exclude=output/ $(REMOTE_HOST):$(REMOTE_PROJ)/ ./

## Remote sweep workflow
# Run a sweep on mililab and bring back its outputs.  Cache is unified:
# remote runs.db is merged into local runs.db so cell parquets remain
# discoverable via `neuro list`.
pull-remote-output: ## Pull output/ from mililab and merge runs.db into local cache
	rsync $(RSYNC_FLAGS) -r --exclude='*.tmp' --exclude='runs.db' \
		$(REMOTE_HOST):$(REMOTE_PROJ)/output/ ./output/
	rsync $(RSYNC_FLAGS) $(REMOTE_HOST):$(REMOTE_PROJ)/output/runs.db /tmp/neuro_remote_runs.db || true
	if [ -f /tmp/neuro_remote_runs.db ]; then \
		uv run neuro cache merge /tmp/neuro_remote_runs.db; \
		rm -f /tmp/neuro_remote_runs.db; \
	fi

sweep-remote: ## Push, run `neuro sweep run $(ARGS)` on mililab, pull results. Usage: make sweep-remote ARGS="--x-var r_pre ..."
	$(MAKE) push-remote
	ssh -t $(REMOTE_HOST) "cd $(REMOTE_PROJ) && uv run neuro sweep run $(ARGS)"
	$(MAKE) pull-remote-output

remote-shell: ## SSH into mililab and cd to the project directory
	ssh -t $(REMOTE_HOST) "cd $(REMOTE_PROJ) && exec bash -l"
