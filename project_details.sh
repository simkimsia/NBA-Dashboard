#!/bin/bash

# This Script is From ChatGPT

RED='\033[0;31m'
format_gitlog() {
	git log "$1" -n 5 --oneline | awk '{print; print ""}'
}
# Get the current git branch
current_branch=$(git branch --show-current)

# Get the git log of the main branch (assuming 'main' is the default branch)
main_log=$(format_gitlog main)

# Get the git log of the current branch
current_log=$(format_gitlog "$current_branch")

# Get the issues from the upstream repository (requires GitHub CLI)
# Replace 'your-repo' with your repository's name and 'your-username' with your GitHub username
upstream_issues=$(gh issue list)


# Display the information
echo "$RED"
echo "You are currently on branch: {$current_branch}"
echo "--------------------------------------------------------------------------------"
echo
echo "Main branch log:"
echo "$main_log"
echo "--------------------------------------------------------------------------------"
echo
echo "Current branch log:"
echo "$current_log"
echo "--------------------------------------------------------------------------------"
echo
echo "Issues from upstream repo:"
echo "$upstream_issues"

