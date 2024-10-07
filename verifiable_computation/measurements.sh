#!/bin/bash

# Function to measure performance
measure_performance() {
    local task_name=$1
    shift
    local args=("$@")

    echo "Measuring performance for: $task_name"

    # CSV header (only for the first task)
    if [ ! -f "${task_name}_results.csv" ]; then
        echo "Input Size,Execution Time (ms),Peak Memory (MB)" > "${task_name}_results.csv"
    fi

    # Run the command for each input size
    for size in "${SIZES[@]}"; do
        echo "Running with input size: $size"

        # Capture start time in milliseconds
        START_TIME=$(($(date +%s%N) / 1000000))
        START_TIME22=$(date )
        
        echo "START_TIME: $START_TIME22"


        OUTPUT=$(
            /usr/bin/time -f "\nExecution time: %E\nPeak memory: %M KB" "$BINARY" "${args[@]}"  2>&1
        )

        # Capture end time in milliseconds
        END_TIME=$(($(date +%s%N) / 1000000))

        # Calculate execution time in milliseconds
        EXEC_TIME_MS=$((END_TIME - START_TIME))

        # Extract memory usage
        MEMORY=$(echo "$OUTPUT" | grep "Peak memory:" | awk '{print $3}')

        # Convert memory to MB and round to 2 decimal places
        MEMORY_MB=$(echo "scale=2; $MEMORY / 1024" | bc)

        # Print the results
        echo "Execution time: $EXEC_TIME_MS milliseconds"
        echo "Peak memory usage: $MEMORY_MB MB"

        # Append to CSV
        echo "$size,$EXEC_TIME_MS,$MEMORY_MB" >> "${task_name}_results.csv"

        # Print the program output
        echo -e "\nProgram output:"
        echo "$OUTPUT" | sed '/Execution time:/d' | sed '/Peak memory:/d'

        echo -e "\n------------------------\n"
    done

    echo "Results for $task_name have been saved to ${task_name}_results.csv"
}

# Check if a binary name is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_binary>"
    exit 1
fi

BINARY="$1"

# Check if the binary exists and is executable
if [ ! -x "$BINARY" ]; then
    echo "Error: $BINARY is not executable or does not exist."
    exit 1
fi


# Measure performance for different tasks
measure_performance "create_keys" --create-keys
SIZES=(1)
measure_performance "training_tee_20_gm_tee"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 20 --read-ds-num-features 1 --type-global-model 1
measure_performance "training_tee_40_gm_tee"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 40 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_tee_80_gm_tee"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 80 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_tee_160_gm_tee"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 160 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_tee_320_gm_tee"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 320 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_tee_640_gm_tee"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 640 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_tee_1280_gm_tee"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 1280 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_tee_2560_gm_tee"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 2560 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_tee_5120_gm_tee"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 5120 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_tee_10240_gm_tee"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 10240 --read-ds-num-features 1  --type-global-model 1

measure_performance "training_tee_20_gm_zkvm"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 20 --read-ds-num-features 1 --type-global-model 2
measure_performance "training_tee_40_gm_zkvm"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 40 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_tee_80_gm_zkvm"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 80 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_tee_160_gm_zkvm"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 160 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_tee_320_gm_zkvm"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 320 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_tee_640_gm_zkvm"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 640 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_tee_1280_gm_zkvm"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 1280 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_tee_2560_gm_zkvm"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 2560 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_tee_5120_gm_zkvm"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 5120 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_tee_10240_gm_zkvm"  --previous-training-type 1 --current-compute-type 1 --training --num-aggregations 1 --read-ds-num-samples 10240 --read-ds-num-features 1  --type-global-model 2

measure_performance "training_zkvm_20_gm_tee"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 20 --read-ds-num-features 1  --type-global-model 1 
measure_performance "training_zkvm_40_gm_tee"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 40 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_zkvm_80_gm_tee"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 80 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_zkvm_160_gm_tee"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 160 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_zkvm_320_gm_tee"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 320 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_zkvm_640_gm_tee"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 640 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_zkvm_1280_gm_tee"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 1280 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_zkvm_2560_gm_tee"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 2560 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_zkvm_5120_gm_tee"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 5120 --read-ds-num-features 1  --type-global-model 1
measure_performance "training_zkvm_10240_gm_tee"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 10240 --read-ds-num-features 1  --type-global-model 1

measure_performance "training_zkvm_20_gm_zkvm"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 20 --read-ds-num-features 1  --type-global-model 2 
measure_performance "training_zkvm_40_gm_zkvm"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 40 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_zkvm_80_gm_zkvm"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 80 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_zkvm_160_gm_zkvm"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 160 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_zkvm_320_gm_zkvm"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 320 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_zkvm_640_gm_zkvm"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 640 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_zkvm_1280_gm_zkvm"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 1280 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_zkvm_2560_gm_zkvm"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 2560 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_zkvm_5120_gm_zkvm"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 5120 --read-ds-num-features 1  --type-global-model 2
measure_performance "training_zkvm_10240_gm_zkvm"  --previous-training-type 1 --current-compute-type 2 --training --num-aggregations 1 --read-ds-num-samples 10240 --read-ds-num-features 1  --type-global-model 2

measure_performance "aggregate_from_tee_inside_tee_2"  --previous-training-type 1 --current-compute-type 1 --aggregate --num-aggregations 2 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_tee_5"  --previous-training-type 1 --current-compute-type 1 --aggregate --num-aggregations 5 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_tee_10"  --previous-training-type 1 --current-compute-type 1 --aggregate --num-aggregations 10 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_tee_20"  --previous-training-type 1 --current-compute-type 1 --aggregate --num-aggregations 20 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_tee_50"  --previous-training-type 1 --current-compute-type 1 --aggregate --num-aggregations 50 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_tee_100"  --previous-training-type 1 --current-compute-type 1 --aggregate --num-aggregations 100 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_tee_200"  --previous-training-type 1 --current-compute-type 1 --aggregate --num-aggregations 200 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_tee_500"  --previous-training-type 1 --current-compute-type 1 --aggregate --num-aggregations 500 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_tee_800"  --previous-training-type 1 --current-compute-type 1 --aggregate --num-aggregations 800 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1

measure_performance "aggregate_from_tee_inside_zkvm_2"  --previous-training-type 1 --current-compute-type 2 --aggregate --num-aggregations 2 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_zkvm_5"  --previous-training-type 1 --current-compute-type 2 --aggregate --num-aggregations 5 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_zkvm_10"  --previous-training-type 1 --current-compute-type 2 --aggregate --num-aggregations 10 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_zkvm_20"  --previous-training-type 1 --current-compute-type 2 --aggregate --num-aggregations 20 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_zkvm_50"  --previous-training-type 1 --current-compute-type 2 --aggregate --num-aggregations 50 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_zkvm_100"  --previous-training-type 1 --current-compute-type 2 --aggregate --num-aggregations 100 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_zkvm_200"  --previous-training-type 1 --current-compute-type 2 --aggregate --num-aggregations 200 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_zkvm_500"  --previous-training-type 1 --current-compute-type 2 --aggregate --num-aggregations 500 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_tee_inside_zkvm_800"  --previous-training-type 1 --current-compute-type 2 --aggregate --num-aggregations 800 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1

measure_performance "aggregate_from_zkvm_inside_tee_2"  --previous-training-type 2 --current-compute-type 1 --aggregate --num-aggregations 2 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_tee_5"  --previous-training-type 2 --current-compute-type 1 --aggregate --num-aggregations 5 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_tee_10"  --previous-training-type 2 --current-compute-type 1 --aggregate --num-aggregations 10 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_tee_20"  --previous-training-type 2 --current-compute-type 1 --aggregate --num-aggregations 20 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_tee_50"  --previous-training-type 2 --current-compute-type 1 --aggregate --num-aggregations 50 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_tee_100"  --previous-training-type 2 --current-compute-type 1 --aggregate --num-aggregations 100 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_tee_200"  --previous-training-type 2 --current-compute-type 1 --aggregate --num-aggregations 200 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_tee_500"  --previous-training-type 2 --current-compute-type 1 --aggregate --num-aggregations 500 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_tee_800"  --previous-training-type 2 --current-compute-type 1 --aggregate --num-aggregations 800 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1

measure_performance "aggregate_from_zkvm_inside_zkvm_2"  --previous-training-type 2 --current-compute-type 2 --aggregate --num-aggregations 2 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_zkvm_5"  --previous-training-type 2 --current-compute-type 2 --aggregate --num-aggregations 5 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_zkvm_10"  --previous-training-type 2 --current-compute-type 2 --aggregate --num-aggregations 10 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_zkvm_20"  --previous-training-type 2 --current-compute-type 2 --aggregate --num-aggregations 20 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_zkvm_50"  --previous-training-type 2 --current-compute-type 2 --aggregate --num-aggregations 50 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_zkvm_100"  --previous-training-type 2 --current-compute-type 2 --aggregate --num-aggregations 100 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_zkvm_200"  --previous-training-type 2 --current-compute-type 2 --aggregate --num-aggregations 200 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_zkvm_500"  --previous-training-type 2 --current-compute-type 2 --aggregate --num-aggregations 500 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
measure_performance "aggregate_from_zkvm_inside_zkvm_800"  --previous-training-type 2 --current-compute-type 2 --aggregate --num-aggregations 800 --read-ds-num-samples 20 --read-ds-num-features 1   --type-global-model 1
