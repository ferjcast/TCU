#!/bin/bash


cargo build --release

cp target/release/host host_final

current_dir=$(pwd)
executable_path="$current_dir/host_final"

./measurements.sh $executable_path