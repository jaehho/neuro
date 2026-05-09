SHELL := /bin/bash

.SILENT:
.DEFAULT_GOAL := help
.ONESHELL:

.PHONY: help remote remote-shell

REMOTE_HOST := mililab
REMOTE_PROJ := /home/jaeho/neuro
REMOTE_TMUX := neuro

## General
help: ## Show this help message
	echo "Available targets:"
	echo "=================="
	grep -hE '(^[a-zA-Z_%-]+:.*?## .*$$|^## )' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; \
		     /^## / {gsub("^## ", ""); print "\n\033[1;35m" $$0 "\033[0m"}; \
		     /^[a-zA-Z_%-]+:/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## Remote compute (mililab)
# `remote` ships local commits via git, runs CMD inside a detached tmux
# session on $(REMOTE_HOST), attaches a TTY for live streaming, and
# rsyncs output/ back when the job ends.
#
# All remote commands are routed through bash (via `bash -c '...'` for
# one-liners or piped to bash via stdin for multi-line scripts) so they
# work regardless of the user's login shell on $(REMOTE_HOST).  The
# inner tmux session command is also `bash -c '...'` so the inner shell
# is bash even if tmux's default-shell on the host is fish.
#
# Resilience: SSH drop / Ctrl-B d / Ctrl-C of make all leave the tmux
# session running.  Re-run `make remote` (CMD optional when a session
# is already alive) to re-attach and resume polling.

remote: ## Push, run CMD on $(REMOTE_HOST) in tmux:$(REMOTE_TMUX), stream live, pull outputs. CMD optional if session running.
	set -euo pipefail
	if ssh $(REMOTE_HOST) "bash -c 'tmux has-session -t $(REMOTE_TMUX) 2>/dev/null'"; then
		echo ">>> existing tmux:$(REMOTE_TMUX) on $(REMOTE_HOST); reattaching (CMD ignored)"
	else
		if [ -z "$(CMD)" ]; then
			echo "Usage: make remote CMD=\"<command to run on $(REMOTE_HOST)>\""
			echo "       (CMD is only optional when a tmux:$(REMOTE_TMUX) session is already running)"
			exit 2
		fi
		if [ -n "$$(git status --porcelain)" ]; then
			echo "ERROR: local working tree is dirty. Commit (or stash) first."
			git status --short
			exit 1
		fi
		echo ">>> checking $(REMOTE_HOST) working tree"
		remote_dirty=$$(ssh $(REMOTE_HOST) "bash -c 'cd $(REMOTE_PROJ) && git status --porcelain'")
		if [ -n "$$remote_dirty" ]; then
			echo "ERROR: $(REMOTE_HOST):$(REMOTE_PROJ) working tree is dirty:"
			echo "$$remote_dirty"
			echo "Clean it via 'make remote-shell' (commit, stash, or git restore)."
			exit 1
		fi
		echo ">>> pushing local commits"
		git push
		echo ">>> launching tmux:$(REMOTE_TMUX) (git pull && $(CMD))"
		ssh $(REMOTE_HOST) bash <<-'EOF'
			tmux new-session -d -s $(REMOTE_TMUX) "bash -c 'cd $(REMOTE_PROJ) && git pull --ff-only && ($(CMD)); echo \$$? > /tmp/$(REMOTE_TMUX)-rc; sleep 2'"
		EOF
		if ! ssh $(REMOTE_HOST) "bash -c 'tmux has-session -t $(REMOTE_TMUX) 2>/dev/null'"; then
			echo "ERROR: failed to launch tmux:$(REMOTE_TMUX) on $(REMOTE_HOST)"
			ssh $(REMOTE_HOST) "bash -c 'tmux ls 2>/dev/null; cat /tmp/$(REMOTE_TMUX)-rc 2>/dev/null'" || true
			exit 1
		fi
	fi
	echo ">>> attaching (Ctrl-B d to detach safely; job continues on $(REMOTE_HOST))"
	set +e
	ssh -t $(REMOTE_HOST) "bash -c 'tmux has-session -t $(REMOTE_TMUX) 2>/dev/null && tmux attach -t $(REMOTE_TMUX) || echo \"(session already ended)\"'"
	set -e
	echo ">>> waiting for tmux:$(REMOTE_TMUX) to finish"
	ssh $(REMOTE_HOST) "bash -c 'while tmux has-session -t $(REMOTE_TMUX) 2>/dev/null; do sleep 10; done'"
	rc=$$(ssh $(REMOTE_HOST) "bash -c 'cat /tmp/$(REMOTE_TMUX)-rc 2>/dev/null || echo 1'")
	echo ">>> remote exited $$rc; syncing output/ back"
	rsync -av --update --exclude='*.tmp' --exclude='runs.db' \
		$(REMOTE_HOST):$(REMOTE_PROJ)/output/ ./output/ || true
	if rsync -a $(REMOTE_HOST):$(REMOTE_PROJ)/output/runs.db /tmp/neuro_remote_runs.db 2>/dev/null; then
		uv run neuro cache merge /tmp/neuro_remote_runs.db
		rm -f /tmp/neuro_remote_runs.db
	fi
	exit $$rc

remote-shell: ## SSH into $(REMOTE_HOST) and cd to the project directory
	ssh -t $(REMOTE_HOST) "cd $(REMOTE_PROJ) && exec bash -l"
