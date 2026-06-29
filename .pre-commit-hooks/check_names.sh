#!/bin/bash

# Check if repository is on berenslab
remote_url=$(git config --get remote.origin.url)
if [[ $remote_url == *"berenslab"* ]]; then
    echo "Skipping name check for berenslab repository"
    exit 0
fi

# Initialize a flag to track if any violations are found
violations_found=0

for notebook in "$@"; do
    # Check if the notebook contains "Student names:"
    if grep -q "Student names:" "$notebook"; then
        # If it does, check if it also contains "FILL IN YOUR NAMES HERE"
        if grep -q "FILL IN YOUR NAMES HERE" "$notebook"; then
            echo "Error: $notebook contains 'FILL IN YOUR NAMES HERE'. Please add your name."
            violations_found=1
        fi
    else
        echo "Warning: $notebook does not contain 'Student names:' section."
        violations_found=1
    fi
done

# If any violations were found, exit with a non-zero status
if [ $violations_found -ne 0 ]; then
    exit 1
fi

# If we've made it here, all checks passed
echo "All notebooks checked. No issues found."
exit 0