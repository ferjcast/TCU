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

use risc0_zkvm::{guest::env, serde};
use smartcore::linalg::basic::{arrays::Array2, matrix::DenseMatrix};
use risc0_zkvm::sha::Digest;

fn main() {
    // n and e are the public modulus and exponent respectively.
    // x value that will be kept private.risc0_zkvm::sha::Digest
    let (n, a, image_id): (DenseMatrix<u8>, DenseMatrix<u8>, risc0_zkvm::sha::Digest) = env::read();
    
    // Verify that n has a known factorization.
    env::verify(image_id, &serde::to_vec(&n).unwrap()).unwrap();

    
    // Set `a` to the first value plus 1
    let z = n.matmul(&a);
    // Commit n, e, and x^e mod n.
    env::commit(&(z));
}
