#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Usage:
#    # Do work and commit your work.

#    # Format files that differ from the determined merge base (upstream/main, origin/main, or local main).
#    bash format.sh

#    # Commit changed files with message 'Run yapf and ruff'
#
#
# YAPF + Ruff + Codespell. This script formats, lints, and spell-checks changed files
# based on the merge-base with upstream/main, origin/main, or local main.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

# --- Tool Version Checks ---
YAPF_VERSION=$(yapf --version | awk '{print $2}')
RUFF_VERSION=$(ruff --version | awk '{print $2}')
# Handle potential variations in codespell version output
CODESPELL_RAW_VERSION=$(codespell --version)
if [[ "$CODESPELL_RAW_VERSION" == codespell* ]]; then
    CODESPELL_VERSION=$(echo "$CODESPELL_RAW_VERSION" | awk '{print $2}') # Assuming format "codespell x.y.z"
else
    CODESPELL_VERSION="$CODESPELL_RAW_VERSION" # Use as is if format is different
fi


# params: tool name, tool version, required version from file
tool_version_check() {
    local tool_name=$1
    local installed_version=$2
    local requirement_line
    local required_version
    
    # Find the requirement line robustly (handles == and ===)
    requirement_line=$(grep "^${tool_name}[=]=" requirements-dev.txt) || requirement_line=$(grep "^${tool_name}=" requirements-dev.txt)

    if [ -z "$requirement_line" ]; then
        echo "Warning: Could not find requirement for '$tool_name' in requirements-dev.txt."
        return # Don't exit, just warn if requirement is missing
    fi

    # Extract version after the last '='
    required_version=$(echo "$requirement_line" | rev | cut -d'=' -f1 | rev)

    # Special handling for codespell if it only prints version number
    if [[ "$tool_name" == "codespell" ]] && [[ "$installed_version" != codespell* ]]; then
         # If installed_version is just the number, compare directly
         if [[ "$installed_version" != "$required_version" ]]; then
            echo "Wrong $tool_name version installed: $required_version is required, not $installed_version."
            echo "Requirement line: $requirement_line"
            exit 1
         fi
    else
        # Standard comparison (handles 'tool x.y.z' or just 'x.y.z' if awk worked)
        # Extract version number from installed_version if needed
        local installed_version_num=$installed_version
        if [[ "$installed_version" == ${tool_name}* ]]; then
            installed_version_num=$(echo "$installed_version" | awk '{print $2}')
        fi

        if [[ "$installed_version_num" != "$required_version" ]]; then
            echo "Wrong $tool_name version installed: $required_version is required, not $installed_version_num (from '$installed_version')."
            echo "Requirement line: $requirement_line"
            exit 1
        fi
    fi
}

tool_version_check "yapf" "$YAPF_VERSION"
tool_version_check "ruff" "$RUFF_VERSION"
tool_version_check "codespell" "$CODESPELL_VERSION"

# --- Determine Merge Base ---
# Define the upstream repository URL to compare against
UPSTREAM_REPO="https://github.com/microsoft/BitBLAS"
MERGEBASE="" # Initialize MERGEBASE variable

echo "Determining merge base for diff..."

# 1. Try to compare directly with the main branch of the upstream repository
if git ls-remote --exit-code "$UPSTREAM_REPO" main &>/dev/null; then
    echo "Attempting to find merge base with upstream: $UPSTREAM_REPO main"
    MERGEBASE_CMD_OUTPUT=$(git fetch "$UPSTREAM_REPO" main --quiet --no-tags 2>/dev/null && git merge-base FETCH_HEAD HEAD)
    FETCH_STATUS=$?
    if [ $FETCH_STATUS -eq 0 ] && [ -n "$MERGEBASE_CMD_OUTPUT" ]; then
        MERGEBASE="$MERGEBASE_CMD_OUTPUT"
        echo "Successfully found merge base with upstream: $MERGEBASE"
    else
        echo "Warning: Could not determine merge base with $UPSTREAM_REPO main (fetch/merge-base failed or no common ancestor). Falling back..."
    fi
fi

# 2. If MERGEBASE could not be obtained from upstream, try using origin/main
if [ -z "$MERGEBASE" ] && git show-ref --verify --quiet refs/remotes/origin/main; then
    echo "Falling back to merge base with origin/main"
    BASE_BRANCH="origin/main"
    MERGEBASE_CMD_OUTPUT=$(git merge-base "$BASE_BRANCH" HEAD)
    MERGEBASE_STATUS=$?
    if [ $MERGEBASE_STATUS -eq 0 ] && [ -n "$MERGEBASE_CMD_OUTPUT" ]; then
        MERGEBASE="$MERGEBASE_CMD_OUTPUT"
        echo "Successfully found merge base with $BASE_BRANCH: $MERGEBASE"
    else
         echo "Warning: Could not determine merge base with $BASE_BRANCH. Falling back..."
    fi
fi

# 3. If even origin/main doesn't work, try using the local main branch
if [ -z "$MERGEBASE" ]; then
    echo "Falling back to merge base with local main"
    BASE_BRANCH="main"
    if git show-ref --verify --quiet "refs/heads/$BASE_BRANCH"; then
        MERGEBASE_CMD_OUTPUT=$(git merge-base "$BASE_BRANCH" HEAD)
        MERGEBASE_STATUS=$?
        if [ $MERGEBASE_STATUS -eq 0 ] && [ -n "$MERGEBASE_CMD_OUTPUT" ]; then
           MERGEBASE="$MERGEBASE_CMD_OUTPUT"
           echo "Successfully found merge base with $BASE_BRANCH: $MERGEBASE"
        else
           echo "Warning: Could not determine merge base with local $BASE_BRANCH."
        fi
    else
         echo "Warning: Local branch '$BASE_BRANCH' not found."
    fi
fi

# 4. Final check for MERGEBASE
if [ -z "$MERGEBASE" ]; then
    echo "Error: Could not determine a suitable merge base. Unable to proceed with diffing changed files."
    exit 1
fi

echo "Using final merge base: $MERGEBASE"
# --- Merge Base Determined ---


# --- YAPF Formatting ---
echo '--- bitblas yapf: Check Start ---'

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)
YAPF_EXCLUDES=(
    '--exclude' 'build/**'
)

# Format specified files
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" "$@"
}

# Format files that differ from the determined merge base.
format_changed() {
    # Use the globally determined $MERGEBASE
    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        echo "Running yapf on changed Python files..."
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
             yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
    else
        echo "No Python files changed according to yapf."
    fi
}

# Format all files
format_all() {
    echo "Running yapf on all Python files..."
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" .
}

# YAPF Execution Logic
if [[ "$1" == '--files' ]]; then
   format "${@:2}"
elif [[ "$1" == '--all' ]]; then
   format_all
else
   format_changed
fi
echo '--- bitblas yapf: Done ---'


# --- Codespell Check ---
echo '--- bitblas codespell: Check Start ---'

# Check spelling of specified files
spell_check() {
    codespell "$@"
}

# Check spelling based on pyproject.toml config (usually checks all relevant files)
spell_check_all(){
  echo "Running codespell based on pyproject.toml..."
  codespell --toml pyproject.toml
}

# Check spelling of files that differ from the determined merge base.
spell_check_changed() {
    # Use the globally determined $MERGEBASE
    # Check Python and potentially other relevant text files (adjust patterns as needed)
    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' '*.md' '*.rst' &>/dev/null; then
        echo "Running codespell on changed text files..."
        # Note: Consider filtering for files codespell actually handles if needed
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' '*.md' '*.rst' | xargs \
             codespell --quiet-level 3 # Adjust quiet level as needed
    else
        echo "No relevant text files changed according to codespell."
    fi
}

# Codespell Execution Logic
if [[ "$1" == '--files' ]]; then
   spell_check "${@:2}"
elif [[ "$1" == '--all' ]]; then
   spell_check_all
else
   spell_check_changed
fi
echo '--- bitblas codespell: Done ---'


# --- Ruff Linting ---
echo '--- bitblas ruff: Check Start ---'

# Lint specified files
lint() {
    ruff check "$@"
}

# Lint files that differ from the determined merge base.
lint_changed() {
    # Use the globally determined $MERGEBASE
    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        echo "Running ruff check on changed Python files..."
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
             ruff check
    else
        echo "No Python files changed according to ruff."
    fi
}

# Ruff Execution Logic
if [[ "$1" == '--files' ]]; then
   lint "${@:2}"
elif [[ "$1" == '--all' ]]; then
   echo "Running ruff check on specified directories..."
   # Adjust directories as needed for your project structure
   lint BitBLAS tests # Assuming these are the main directories
else
   lint_changed
fi
echo '--- bitblas ruff: Done ---'

# --- Final Check for Changes ---
# Check if yapf (or potentially other tools if they modify files) made changes
if ! git diff --quiet &>/dev/null; then
    echo
    echo '-----------------------------------------------------------------------'
    echo 'Detected changes made by the formatting/linting tools.'
    echo 'Please review and stage these changes before committing:'
    echo '-----------------------------------------------------------------------'
    echo
    git --no-pager diff --color=always # Show colored diff directly
    echo
    echo '-----------------------------------------------------------------------'
    echo 'Exiting with status 1 due to needed changes.'
    echo '-----------------------------------------------------------------------'
    exit 1
fi

echo
echo '--- bitblas: All checks passed ---'
exit 0