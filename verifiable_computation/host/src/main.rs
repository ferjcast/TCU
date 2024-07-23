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

// The serialized trained model and input data are embedded from files
// corresponding paths listed below. Alternatively, the model can be trained in
// the host and/or data can be manually inputted as a smartcore DenseMatrix. If
// this approach is desired, be sure to import the corresponding SmartCore
// modules and serialize the model and data to byte arrays before transfer to
// the guest.

//use risc0_zkvm::{default_prover, ExecutorEnv};

//use csv::WriterBuilder;
//use jemalloc_ctl::{epoch, stats};

#![allow(deprecated)]
#[allow(unused_variables)]
use anyhow::Context;
use hex;
use risc0_binfmt; //::{MemoryImage, Program};

use serde::{Deserialize, Serialize};
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;
//use smartcore_ml_methods::{ML_TEMPLATE_ELF, ML_TEMPLATE_ID};
use k256::ecdsa::{signature::Signer, Signature, SigningKey, VerifyingKey};
use k256::ecdsa::signature::Verifier;
use k256::sha2::{Digest, Sha256};
use rand_core::OsRng;
use std::fs::File;
use std::io::{Read, Write};

//use std::fs::File;
//use std::io::Write;
use bincode;
//use chrono::Local;
use risc0_zkvm::ExecutorEnv;
use risc0_zkvm::Receipt;

//use std::fs::OpenOptions;
use risc0_zkvm::default_prover;
use std::{error::Error, fs};
use clap::Parser;

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// size of square matrix
    #[arg(short, long, default_value_t = 2)]
    n_size: usize,

    /// multiplication proving, type depends on tee or zkvm
    #[arg(short, long, default_value_t = false)]
    proving: bool,

    /// tee = 1, zkvm = 2
    #[arg(short, long, default_value_t = 2)]
    verifying_type: usize,


    /// tee = 1, zkvm = 2
    #[arg(short = 't', long, default_value_t = 2)]
    compute_type: usize,


    /// create signing keys
    #[arg(short, long, default_value_t = false)]
    create_keys: bool,

}

#[derive(Serialize, Deserialize)]
struct SignedMatrix {
    matrix: Vec<u8>, // Serialized matrix
    hash: String,
    signature: String,
}

fn create_random_matrix(n: usize, seed: Option<u64>) -> DenseMatrix<u8> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    
    let data: Vec<u8> = (0..n*n).map(|_| rng.gen::<u8>()).collect();
    DenseMatrix::new(n, n, data, false)
}

fn compute_hash(matrix: &DenseMatrix<u8>) -> Vec<u8> {
    let mut hasher = Sha256::new();

    // Convert matrix data to bytes and update hasher
    for &val in matrix.iter() {
        hasher.update(val.to_le_bytes());
    }

    // Consume the hasher and retrieve the result
    hasher.finalize().to_vec()
}

fn store_keys(signing_key: &SigningKey, file_path: &str) -> std::io::Result<()> {
    let key_bytes = signing_key.to_bytes();
    let mut file = File::create(file_path)?;
    file.write_all(&key_bytes)?;
    Ok(())
}

fn read_signing_key(file_path: &str) -> std::io::Result<SigningKey> {
    let mut file = File::open(file_path)?;
    let mut key_bytes = [0u8; 32];
    file.read_exact(&mut key_bytes)?;
    Ok(SigningKey::from_bytes((&key_bytes).into()).expect("Invalid key"))
}

fn store_verifying_key(verifying_key: &VerifyingKey, file_path: &str) -> std::io::Result<()> {
    let binding = verifying_key.to_encoded_point(false);
    let key_bytes = binding.as_bytes();
    let mut file = File::create(file_path)?;
    file.write_all(key_bytes)?;
    Ok(())
}
fn read_verifying_key(file_path: &str) -> std::io::Result<VerifyingKey> {
    let mut file = File::open(file_path)?;
    let mut key_bytes = Vec::new();
    file.read_to_end(&mut key_bytes)?;
    Ok(VerifyingKey::from_sec1_bytes(&key_bytes).expect("Invalid verifying key"))
}
fn verify_signature(verifying_key: &VerifyingKey, message: &[u8], signature: &Signature) -> bool {
    verifying_key.verify(message, signature).is_ok()
}
fn store_signed_matrix(matrix: &DenseMatrix<u8>, signature: &Signature, file_path: &str) -> std::io::Result<()> {
    let hash = compute_hash(matrix);
    let signed_matrix = SignedMatrix {
        matrix: bincode::serialize(matrix).expect("Failed to serialize matrix"),
        hash: hex::encode(hash),
        signature: base64::encode(signature.to_bytes()),
    };
    let json = serde_json::to_string(&signed_matrix)?;
    let mut file = File::create(file_path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}
fn read_signed_matrix(file_path: &str) -> std::io::Result<SignedMatrix> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let signed_matrix: SignedMatrix = serde_json::from_str(&contents)?;
    Ok(signed_matrix)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let n_size = args.n_size;
    let proving = args.proving;
    let verifying_type = args.verifying_type;
    let compute_type = args.compute_type;
    let create_keys = args.create_keys;

    if create_keys{
        println!("Creating keys for signing and verification");

        let signing_key = SigningKey::random(&mut OsRng); // Serialize with `::to_bytes()`
        let verifying_key = signing_key.verifying_key();
        store_keys(&signing_key, "./keys/signing_key.bin")?;
        store_verifying_key(&verifying_key, "./keys/verifying_key.bin")?;
        println!("Keys created and stored at ./keys/");
    }

    if compute_type == 1 {
        println!("Working on TEE Workflow");

        if proving {        
            println!("Proving TEE Workflow");
            println!("Square Matrix size {}", n_size);
            

            let x: DenseMatrix<u8> = create_random_matrix(n_size, Some(12345));
            let y: DenseMatrix<u8> = create_random_matrix(n_size, Some(12345));
            let unwrap_m: DenseMatrix<u8> = x.matmul(&y); 
            //println!("{}", unwrap_m);
            let random_matrix_hash: Vec<u8> = compute_hash(&unwrap_m);
            let read_signing_key = read_signing_key("./keys/signing_key.bin")?;
            let signing_key = read_signing_key;
            let message = &random_matrix_hash; //b"This is a message that will be signed, and verified within the zkVM";
            let signature: Signature = signing_key.sign(message);
            let file_path = format!("./verifiable_artifacts/signed_messages/signed_matrix_{}.json", n_size);

            store_signed_matrix(&unwrap_m, &signature, &file_path)?;
            println!("Multiplication, Signature and Storing done");

        }
        if verifying_type == 1{
            println!("Verifying TEE");
            println!("Square Matrix size {}", n_size);
            let file_path = format!("./verifiable_artifacts/signed_messages/signed_matrix_{}.json", n_size);
            let signed_matrix = read_signed_matrix(&file_path)?;
            let read_matrix: DenseMatrix<u8> = bincode::deserialize(&signed_matrix.matrix)
                .expect("Failed to deserialize matrix");
            let read_hash = compute_hash(&read_matrix);
            let hash_valid = hex::encode(read_hash.clone()) == signed_matrix.hash;
            let verifying_key = read_verifying_key("./keys/verifying_key.bin")?;

            // Verify the signature
            let read_signature = Signature::from_slice(&base64::decode(signed_matrix.signature).expect("Invalid base64 signature"))
                .expect("Invalid signature");
            let signature_valid =verify_signature(&verifying_key, &read_hash, &read_signature);

            println!("Hash is valid: {}", hash_valid);
            println!("Signature is valid: {}", signature_valid);
            println!("Verification complete");
        
        }
        else if verifying_type == 2{
            println!("Verifying ZKVM");   
            println!("Square Matrix size {}", n_size);      
            
            let elf_path = "./elf_binaries/matrix_multiplication_proving"; // target_dir.join(&pkg_name).join(&target.name);
            let elf = std::fs::read(&elf_path)
                .with_context(|| format!("Failed to read ELF file at path: {}", elf_path))?;
            let image_id = risc0_binfmt::compute_image_id(&elf)?;

            let base_name = "./verifiable_artifacts/receipts/receipt_multiplication_size_";
            let receipt_path = format!("{}{}", base_name, n_size);
            let receipt: Receipt = bincode::deserialize(&fs::read(receipt_path)?)?;
            
            receipt.verify(image_id).expect(
                        "Code you have proven should successfully verify; did you specify the correct image ID?",
                    );
            println!("Matrix Multiplication Proof Verified");
            
        }
    }
    else if compute_type == 2{
        println!("Working on ZKVM Workflow");
        println!("Square Matrix size {}", n_size);

        if proving{        
            println!("Proving ZKVM Workflow");
            println!("Square Matrix size {}", n_size);
            let elf_path = "./elf_binaries/matrix_multiplication_proving"; // target_dir.join(&pkg_name).join(&target.name);
            let elf = std::fs::read(&elf_path)
            .with_context(|| format!("Failed to read ELF file at path: {}", elf_path))?;
            let n = n_size;
            let x: DenseMatrix<u8> = create_random_matrix(n_size, Some(12345));
            let y: DenseMatrix<u8> = create_random_matrix(n_size, Some(12345));
            println!("{}", x);
            println!("{}", y);
            let env = ExecutorEnv::builder()
                .write(&x)
                .expect("model failed to serialize")
                .write(&y)
                .expect("data failed to serialize")
                .build()
                .unwrap();
            let prover = default_prover();
            
            let receipt = prover.prove(env, &elf).unwrap().receipt;
            let base_name = "./verifiable_artifacts/receipts/receipt_multiplication_size_";
            let receipt_path = format!("{}{}", base_name, n);
            fs::write(&receipt_path, bincode::serialize(&receipt).unwrap())?;
            println!("Proof Receipt Stored in: {}",receipt_path);


        }
        if verifying_type == 1{
            //Verifies TEE signature inside ZKVM, only generates receipt
            println!("Verifying TEE");
            println!("Square Matrix size {}", n_size);
            let file_path = format!("./verifiable_artifacts/signed_messages/signed_matrix_{}.json", n_size);
            let signed_matrix = read_signed_matrix(&file_path)?;
            let read_matrix: DenseMatrix<u8> = bincode::deserialize(&signed_matrix.matrix)
                .expect("Failed to deserialize matrix");
            let read_hash = compute_hash(&read_matrix);
            let hash_valid = hex::encode(read_hash.clone()) == signed_matrix.hash;
            let verifying_key = read_verifying_key("./keys/verifying_key.bin")?;
            let integer_sample: u8 = 1;

            // Verify the signature
            let read_signature = Signature::from_slice(&base64::decode(signed_matrix.signature).expect("Invalid base64 signature"))
                .expect("Invalid signature");

            let elf_path = "./elf_binaries/matrix_multiplication_tee_verification"; // target_dir.join(&pkg_name).join(&target.name);
            let elf = std::fs::read(&elf_path)
            .with_context(|| format!("Failed to read ELF file at path: {}", elf_path))?;

            let env = ExecutorEnv::builder()
                .write(&(
                    read_matrix,
                    integer_sample,
                    verifying_key.to_encoded_point(true),
                    &read_hash,
                    read_signature,
                ))
                .unwrap()
                .build()
                .unwrap();

            let _receipt = default_prover()
                .prove(env, &elf)
                .unwrap()
                .receipt;
            println!("Matrix Multiplication Signature Verified inside ZKVM");

        }
        else if verifying_type == 2{
            //Verifies TEE signature inside ZKVM, only generates receipt
            println!("Verifying ZKVM");
            println!("Square Matrix size {}", n_size);

            let elf_path = "./elf_binaries/matrix_multiplication_proving"; // target_dir.join(&pkg_name).join(&target.name);
            let elf = std::fs::read(&elf_path)
                .with_context(|| format!("Failed to read ELF file at path: {}", elf_path))?;
            let image_id = risc0_binfmt::compute_image_id(&elf)?;

            let base_name = "./verifiable_artifacts/receipts/receipt_multiplication_size_";
            let receipt_path = format!("{}{}", base_name, n_size);
            let receipt: Receipt = bincode::deserialize(&fs::read(receipt_path)?)?;

            let unwrap_m: DenseMatrix<u8> = receipt.journal.decode().expect(
                "Journal output should deserialize into the same types (& order) that it was written",
            );
            let integer_sample: u8 = 1; 

            let elf_path_composed = "./elf_binaries/matrix_multiplication_zkvm_verification"; // target_dir.join(&pkg_name).join(&target.name);
            let elf_compose = std::fs::read(&elf_path_composed)
            .with_context(|| format!("Failed to read ELF file at path: {}", elf_path_composed))?;

            let env = ExecutorEnv::builder()
                .add_assumption(receipt)
                .write(&(unwrap_m, integer_sample, image_id))
                .unwrap()
                .build()
                .unwrap();

            let _receipt = default_prover()
                .prove(env, &elf_compose)
                .unwrap()
                .receipt;
            println!("Matrix Multiplication Receipt Verified inside ZKVM");

            /* If want to verify also the final composed proof
            let image_id_compose = risc0_binfmt::compute_image_id(&elf_compose)?;
            _receipt.verify(image_id_compose).expect(
                "Code you have proven should successfully verify; did you specify the correct image ID?",
            );
            println!("Matrix Multiplication Receipt Verified ");
            */



            
        }
    }

    Ok(())
}
