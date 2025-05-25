#!/bin/bash

# Define the problems and algorithms
problems=("BraninCurrin" "DTLZ1" "DTLZ2" "DTLZ3" "DTLZ4" "DTLZ5" "DTLZ7" "Penicillin" "VehicleSafety" "ZDT1" "ZDT2" "ZDT3" "CarSideImpact")

# Loop over each problem and algorithm
for problem in "${problems[@]}"; do
    echo "Submitting job: ${problem}"
    sbatch --job-name="${problem}" --export=problem=$problem run.sh
done