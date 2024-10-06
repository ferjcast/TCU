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

use k256::{
    ecdsa::{signature::Verifier, Signature, VerifyingKey},
    EncodedPoint,
};

use k256::sha2::{Digest, Sha256};

/// Function to aggregate flattened model weights in multiple substeps
fn average_model(data: &Vec<Vec<f64>>) -> Vec<f64> {
    let num_rows = data.len();
    if num_rows == 0 {
        return vec![]; // Return an empty vector if there are no rows
    }
    
    let num_cols = data[0].len();
    let mut averages: Vec<f64> = vec![0.0; num_cols];

    for row in data {
        for (i, &value) in row.iter().enumerate() {
            averages[i] += value; // Sum the values for each column
        }
    }

    // Calculate the average
    for avg in &mut averages {
        *avg /= num_rows as f64; // Divide by the number of rows to get the average
    }

    averages
}


// Function to compute the hash of the model
fn hash_model(model: &Vec<f64>) -> Vec<u8> {
    let mut hasher = Sha256::new();
    for &value in model {
        // Round the value to 2 decimal places
        let rounded_value = (value * 100.0).round() / 100.0;

        // Hash the rounded value as bytes
        hasher.update(&rounded_value.to_le_bytes()); // Use the byte representation directly

        println!("Hashing value: {}", rounded_value); // Print the rounded value for debugging
    }
    hasher.finalize().to_vec()
}



fn main() {
    // Read in is_svm boolean to ensure the correct code block is executed

    // Read the input data into a DenseMatrix.
    let W: Vec<Vec<f64>> = env::read();
    let encoded_verifying_keys: Vec<EncodedPoint> = env::read();
    let messages: Vec<Vec<u8>> = env::read();
    let signatures: Vec<Signature> = env::read();
    let mut valid_signature = true;
    let mut valid_hash = true;

    for i_agg in 0..W.len() {
        let encoded_verifying_key: EncodedPoint = encoded_verifying_keys[i_agg];
        let message: Vec<u8> = messages[i_agg].clone();
        let signature: Signature = signatures[i_agg];

        let verifying_key = VerifyingKey::from_encoded_point(&encoded_verifying_key).unwrap();
        let is_valid = verifying_key.verify(&message, &signature).is_ok();
        valid_signature = valid_signature && is_valid;

        println!("valid_signature {}", is_valid);

        let ds_hash: Vec<u8> = hash_model(&W[i_agg]);

        let mut hash_validity = false;
    
        if ds_hash == message {
            hash_validity = true;

            println!("random_matrix_hash and message are equal.");
        }
        valid_hash = valid_hash && hash_validity;
    }

    //let b: Vec<Array2<f64>> = env::read();
    let model_nn = average_model(&W);

    // We call the predict() function on our trained model to perform inference.

    // This line is optional and can be commented out, but it's useful to see
    // the output of the computation before the proving step begins.
    //println!("answer: {:?}", &z);

    // We commit the output to the journal.
    env::commit(&(model_nn, valid_signature, valid_hash));

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
