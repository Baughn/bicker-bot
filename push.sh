#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

jj bookmark set master -r 'git_head()'
jj git push
