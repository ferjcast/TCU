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

use smartcore::linalg::basic::{arrays::Array2, matrix::DenseMatrix};

fn main() {
    // Read in is_svm boolean to ensure the correct code block is executed

    // Read the input data into a DenseMatrix.
    let x_data: DenseMatrix<u8> = env::read();
    let y_data: DenseMatrix<u8> = env::read();
    let z = x_data.matmul(&y_data);

    // We call the predict() function on our trained model to perform inference.

    // This line is optional and can be commented out, but it's useful to see
    // the output of the computation before the proving step begins.
    //println!("answer: {:?}", &z);

    // We commit the output to the journal.
    env::commit(&z);

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
