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
//use smartcore_ml_methods::ML_TEMPLATE_ID;

use k256::{
    ecdsa::{signature::Verifier, Signature, VerifyingKey},
    EncodedPoint,
};

use k256::sha2::{Digest, Sha256};

fn compute_hash(matrix: &DenseMatrix<u8>) -> Vec<u8> {
    let mut hasher = Sha256::new();

    // Convert matrix data to bytes and update hasher
    for &val in matrix.iter() {
        hasher.update(val.to_le_bytes());
    }

    // Consume the hasher and retrieve the result
    hasher.finalize().to_vec()
}

/*
fn verify_signature(verifying_key: &VerifyingKey, hash: &[u8], signature: &Signature) -> bool {
    verifying_key.verify(hash, signature).is_ok()
} */

fn main() {
    // n and e are the public modulus and exponent respectively.
    // x value that will be kept private.
    let (n, a, encoded_verifying_key, message, signature): (
        DenseMatrix<u8>,
        DenseMatrix<u8>,
        EncodedPoint,
        Vec<u8>,
        Signature,
    ) = env::read();

    let verifying_key = VerifyingKey::from_encoded_point(&encoded_verifying_key).unwrap();
    let is_valid = verifying_key.verify(&message, &signature).is_ok();
    println!("is_valid {}", is_valid);

    //let z = n.iter() + a;
    // Access the first element of the matrix
    // Set `a` to the first value plus 1
    let random_matrix_hash: Vec<u8> = compute_hash(&n);

    let mut hash_validity = false;

    if random_matrix_hash == message {
        hash_validity = true;
        println!("random_matrix_hash and message are equal.");
    }

    let z = n.matmul(&a);
    
    //println!("The value of z is: {}", z);

    // Verify that n has a known factorization.
    //env::verify(ML_TEMPLATE_ID, &serde::to_vec(&n).unwrap()).unwrap();

    // Commit n, e, and x^e mod n.
    env::commit(&(z, is_valid, hash_validity));

    println!(
        "Total cycles for guest code execution 2: {}",
        env::cycle_count()
    );
}
