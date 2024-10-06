// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use risc0_zkvm::guest::env;


/// Function to aggregate flattened model weights in multiple substeps
pub fn aggregate_flattened_weights(flattened_weights: Vec<Vec<f64>>) -> Vec<f64> {
    assert!(!flattened_weights.is_empty(), "There must be at least one model to aggregate.");

    let num_models = flattened_weights.len() as f64;
    let model_length = flattened_weights[0].len();
    let mut aggregated_weights = vec![0.0; model_length];

    // Step 1: Initialize aggregated weights
    initialize_aggregated_weights(&mut aggregated_weights);

    // Step 2: Sum the weights from each model
    sum_weights(&flattened_weights, &mut aggregated_weights);

    // Step 3: Divide by the number of models
    divide_by_num_models(&mut aggregated_weights, num_models);

    aggregated_weights
}

/// Initializes the aggregated weights to zeros.
fn initialize_aggregated_weights(aggregated_weights: &mut Vec<f64>) {
    for weight in aggregated_weights.iter_mut() {
        *weight = 0.0; // Reset to zero (optional since they are initialized to zero already)
    }
}

/// Sums the weights from each model into the aggregated weights.
fn sum_weights(flattened_weights: &Vec<Vec<f64>>, aggregated_weights: &mut Vec<f64>) {
    for weight in flattened_weights {
        for (i, &value) in weight.iter().enumerate() {
            aggregated_weights[i] += value;
        }
    }
}

/// Divides the aggregated weights by the number of models.
fn divide_by_num_models(aggregated_weights: &mut Vec<f64>, num_models: f64) {
    for weight in aggregated_weights.iter_mut() {
        *weight /= num_models;
    }
}




fn main() {
    // Read in is_svm boolean to ensure the correct code block is executed

    // Read the input data into a DenseMatrix.
    let W: Vec<Vec<f64>> = env::read();
    //let b: Vec<Array2<f64>> = env::read();
    let model_nn = aggregate_flattened_weights(W);

    // We call the predict() function on our trained model to perform inference.

    // This line is optional and can be commented out, but it's useful to see
    // the output of the computation before the proving step begins.
    //println!("answer: {:?}", &z);

    // We commit the output to the journal.
    env::commit(&model_nn);

    // Logging the total cycle count is optional, though it's quite useful for benchmarking
    // the various operations in the guest code. env::cycle_count() can be
    // called anywhere in the guest, multiple times. So if we are interested in
    // knowing how many cycles the inference computation takes, we can calculate
    // total cycles before and after model.predict() and the difference between
    // the two values equals the total cycle count for that section of the guest
    // code.
    println!(
        "Total cycles for guest code execution: {}",
        env::cycle_count()
    );
}
