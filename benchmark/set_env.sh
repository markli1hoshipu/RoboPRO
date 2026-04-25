#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# benchmark lives here; the envs package lives in sibling customized_robotwin
export BENCH_ROOT="$SCRIPT_DIR"
export ROBOTWIN_ROOT="$WORKSPACE_ROOT/customized_robotwin"

echo "BENCH_ROOT=$BENCH_ROOT"
echo "ROBOTWIN_ROOT=$ROBOTWIN_ROOT"
