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
//use risc0_zkvm::sha::Digest;
use risc0_zkvm::Journal;


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



fn main() {
    // Read in is_svm boolean to ensure the correct code block is executed

    // Read the input data into a DenseMatrix.
    //let W: Vec<Vec<f64>> = env::read();
    let (journals, image_id): (Vec<Journal>, risc0_zkvm::sha::Digest) = env::read();
    //let (W, h_validity, s_validity, image_id): (Vec<Vec<f64>>, Vec<bool>, Vec<bool>, Vec<risc0_zkvm::sha::Digest>) = env::read();

    let mut list_models =  Vec::new();
    let mut truecity = true;

    for  journal in journals.into_iter() {
        env::verify(image_id, &journal.bytes).unwrap();
        let (flattened_params, hash_validity, is_valid): (Vec<f64>, bool, bool) = journal.decode().unwrap();
        truecity = truecity && hash_validity && is_valid;
        list_models.push(flattened_params);

    }

    let aggregated_model = average_model(&list_models);




    //let b: Vec<Array2<f64>> = env::read();
    //let model_nn = average_model(W);
    //let aa = ux8+1;

    // We call the predict() function on our trained model to perform inference.

    // This line is optional and can be commented out, but it's useful to see
    // the output of the computation before the proving step begins.
    //println!("answer: {:?}", &z);

    // We commit the output to the journal.
    env::commit(&(aggregated_model, truecity));

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
