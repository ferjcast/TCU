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

        OUTPUT=$(
            /usr/bin/time -f "\nExecution time: %E\nPeak memory: %M KB" "$BINARY" "${args[@]}" --n-size "$size" 2>&1
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
SIZES=(3 5 10 20 30 50 100 200 500 1000 2000)
measure_performance "proving_tee" --compute-type 1 --proving --verifying-type 0
SIZES=(3 5 10 20 30 50 100 200)
measure_performance "proving_zkvm" --compute-type 2 --proving --verifying-type 0
SIZES=(3 5 10 20 30 50 100 200 500 1000 2000)
measure_performance "verifying_tee_inside_tee" --compute-type 1 --verifying-type 1
SIZES=(3 5 10 20 30 50 100 200)
measure_performance "verifying_zkvm_inside_tee" --compute-type 1 --verifying-type 2
SIZES=(3 5 10 20 30 50 100 200 500 1000 2000)
measure_performance "verifying_tee_inside_zkvm" --compute-type 2 --verifying-type 1
SIZES=(3 5 10 20 30 50 100 200)
measure_performance "verifying_zkvm_inside_zkvm" --compute-type 2 --verifying-type 2

echo "All performance measurements complete."