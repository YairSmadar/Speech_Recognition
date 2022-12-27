#!/bin/bash

# Update package manager and install necessary packages
sudo apt-get update
sudo apt-get install python3 python3-pip

# List of required libraries
libraries=("glob" "sklearn" "librosa" "torch" "scipy")

pip install numpy==1.21.0
# Loop through required libraries
for library in "${libraries[@]}"; do
  # Check if library is installed
  if ! python3 -c "import $library" &> /dev/null; then
    # Install library if not installed
    pip3 install $library
  fi
done