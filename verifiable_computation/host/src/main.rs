use anyhow::Context;
use bincode::serialize;
use clap::{Parser, Arg};

use k256::EncodedPoint;
use ndarray::{s, Array1, Array2, Axis, Dimension};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_core::OsRng;
use serde::{Deserialize, Serialize};
use tracing_subscriber::fmt::writer::OrElse;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Write};
use k256::ecdsa::{signature::Signer, Signature, SigningKey, VerifyingKey};
use k256::ecdsa::signature::Verifier;
use k256::sha2::{Digest, Sha256};
use std::{env, path};
use std::num::ParseIntError;
use std::fs::OpenOptions; // Add this import
//use aggregation_methods::{AGGREGATION_ELF,AGGREGATION_ID};
//use smartcore_ml_methods::{ML_TEMPLATE_ELF, ML_TEMPLATE_ID};
//use rand::Rng; // Make sure to include the `rand` crate in your Cargo.toml

//zkvm
use risc0_binfmt; //::{MemoryImage, Program};

use risc0_zkvm::{ExecutorEnv, ExecutorEnvBuilder, Journal};
use risc0_zkvm::Receipt;

//use std::fs::OpenOptions;
use risc0_zkvm::default_prover;

/// Generates a fixed dataset and stores it in a CSV file.
use std::error::Error;
use csv::Writer;


use std::any::Any;

#[derive(Serialize, Deserialize)]
struct Dataset {
    inputs: Vec<Vec<f64>>,
    targets: Vec<Vec<f64>>,
    hash: String,
    signature: String,
}

#[derive(Serialize, Deserialize)]
struct ModelMetadata {
    model: Vec<f64>,            // The model parameters
    model_hash: String,         // The hash of the model (base64-encoded)
    signature: String,          // The signature of the model
}
// Trait defining the structure of a layer in the neural network
pub trait NeuralLayer {
    fn forward(&mut self, x: &Vec<f64>) -> Vec<f64>;
    fn backward(&mut self, dL_dy: &Vec<f64>) -> Vec<f64>;
    fn sgd(&mut self, learning_rate: f64);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn clone_layer(&self) -> Box<dyn NeuralLayer>;

    // New method to flatten the layer parameters
    fn flatten(&self) -> Vec<f64>;
}

// ReLU Layer implementation
#[derive(Clone)]
pub struct ReLULayer {
    input: Vec<f64>, // Store the input for backpropagation
}

impl ReLULayer {
    pub fn new() -> Self {
        ReLULayer {
            input: Vec::new(),
        }
    }
}

impl NeuralLayer for ReLULayer {
    fn forward(&mut self, x: &Vec<f64>) -> Vec<f64> {
        self.input = x.clone(); // Store input for backpropagation
        x.iter().map(|&val| val.max(0.0)).collect()
    }

    fn backward(&mut self, dL_dy: &Vec<f64>) -> Vec<f64> {
        dL_dy.iter()
            .zip(self.input.iter())
            .map(|(dy, &x)| if x > 0.0 { *dy } else { 0.0 })
            .collect()
    }

    fn sgd(&mut self, _learning_rate: f64) {
        // No parameters to update for ReLU
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_layer(&self) -> Box<dyn NeuralLayer> {
        Box::new(self.clone())
    }

    fn flatten(&self) -> Vec<f64> {
        vec![] // ReLU has no parameters to flatten
    }
}

// Linear Layer implementation
#[derive(Clone)]
pub struct LinearLayer {
    W: Vec<Vec<f64>>,    // Weight matrix
    b: Vec<f64>,         // Bias vector
    dL_dW: Vec<Vec<f64>>, // Gradient for W
    dL_db: Vec<f64>,     // Gradient for b
    dy_dW: Option<Vec<f64>>, // Input to keep for backpropagation
}

impl NeuralLayer for LinearLayer {
    fn forward(&mut self, x: &Vec<f64>) -> Vec<f64> {
        self.dy_dW = Some(x.clone());

        // Compute Wx + b
        let mut output = vec![0.0; self.b.len()];
        for (i, row) in self.W.iter().enumerate() {
            output[i] = row.iter().zip(x.iter()).map(|(w, x)| w * x).sum::<f64>() + self.b[i];
        }
        output
    }

    fn backward(&mut self, dL_dy: &Vec<f64>) -> Vec<f64> {
        let x = self.dy_dW.as_ref().expect("Need to call forward() first.");
        
        // Compute gradients for W and b
        for i in 0..self.W.len() {
            for j in 0..self.W[i].len() {
                self.dL_dW[i][j] += dL_dy[i] * x[j];
            }
            self.dL_db[i] += dL_dy[i];
        }
    
        // Return gradient for the input (for backpropagation)
        let mut dL_dx = vec![0.0; x.len()];
        for i in 0..x.len() {
            for j in 0..self.W.len() {
                dL_dx[i] += self.W[j][i] * dL_dy[j];
            }
        }
    
        dL_dx
    }

    fn sgd(&mut self, learning_rate: f64) {
        // Update W and b using gradients
        for (i, row) in self.W.iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                *w -= self.dL_dW[i][j].clamp(-5.0, 5.0) * learning_rate;
            }
            self.b[i] -= self.dL_db[i].clamp(-5.0, 5.0) * learning_rate;
        }
    }

    fn clone_layer(&self) -> Box<dyn NeuralLayer> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn flatten(&self) -> Vec<f64> {
        let mut params = vec![];
        // Flatten weights
        for row in &self.W {
            params.extend(row.clone());
        }
        // Flatten biases
        params.extend(self.b.clone());
        params
    }
}

impl LinearLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let W = vec![vec![0.1; in_features]; out_features];
        let b = vec![0.0; out_features];
        let dL_dW = vec![vec![0.0; in_features]; out_features];
        let dL_db = vec![0.0; out_features];

        LinearLayer {
            W,
            b,
            dL_dW,
            dL_db,
            dy_dW: None,
        }
    }
}

// Mean Squared Error implementation
pub struct MSE {
    dL_dx: Vec<f64>,
}

impl MSE {
    pub fn new() -> Self {
        Self { dL_dx: vec![0.0] }
    }

    fn get_output(x: &Vec<f64>, target: &Vec<f64>) -> f64 {
        let error: f64 = x.iter().zip(target).map(|(a, b)| (a - b).powi(2)).sum();
        error / x.len() as f64
    }

    pub fn forward(&mut self, x: &Vec<f64>, target: &Vec<f64>) -> f64 {
        self.dL_dx = x.iter().zip(target).map(|(a, b)| 2.0 * (a - b)).collect();
        Self::get_output(x, target)
    }

    pub fn backward(&self) -> Vec<f64> {
        self.dL_dx.clone()
    }
}

// Neural Network implementation
pub struct NN {
    layers: Vec<Box<dyn NeuralLayer>>,
}

impl NN {
    pub fn new(in_features: usize, hidden_size: usize, out_features: usize) -> Self {
        let mut layers: Vec<Box<dyn NeuralLayer>> = vec![];
        layers.push(Box::new(LinearLayer::new(in_features, hidden_size)));
        layers.push(Box::new(ReLULayer::new()));
        layers.push(Box::new(LinearLayer::new(hidden_size, out_features)));

        NN { layers }
    }

    pub fn forward(&mut self, batch: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        batch.iter()
            .map(|x| {
                let mut output = x.clone();
                for layer in self.layers.iter_mut() {
                    output = layer.forward(&output);
                }
                output
            })
            .collect() // Collect outputs for each input row
    }

    pub fn backward(&mut self, dy: &Vec<f64>) -> Vec<f64> {
        let mut dy = dy.clone();
        for layer in self.layers.iter_mut().rev() {
            dy = layer.backward(&dy);
        }
        dy
    }

    pub fn train(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, learning_rate: f64) {
        let mut mse = MSE::new();
        let mut total_loss = 0.0;
        
        // Reset gradients before training
        for layer in &mut self.layers {
            match layer.as_any_mut().downcast_mut::<LinearLayer>() {
                Some(linear_layer) => {
                    linear_layer.dL_dW = vec![vec![0.0; linear_layer.W[0].len()]; linear_layer.W.len()];
                    linear_layer.dL_db = vec![0.0; linear_layer.b.len()];
                }
                _ => {}
            }
        }
    
        for (input, target) in inputs.iter().zip(targets) {
            // Forward pass for the current sample
            let output = self.forward(&vec![input.clone()])[0].clone(); // Single input forward
        
            // Compute loss and accumulate
            let loss = mse.forward(&output, target);
            //println!("Step Loss: {}", loss);

            total_loss += loss;
        
            // Compute gradients for the current sample
            let dy = mse.backward();
            self.backward(&dy);
        
            // Update the parameters for each layer immediately (SGD)
            for layer in &mut self.layers {
                layer.sgd(learning_rate);
            }
        }
        
        // Print the average loss after processing all samples
        println!("Average Loss: {}", total_loss / inputs.len() as f64);
    }

    // Function to unflatten a Vec<f64> into the NN structure
    pub fn unflatten(&mut self, params: &Vec<f64>) {
        let mut index = 0;

        for layer in &mut self.layers {
            if let Some(linear_layer) = layer.as_any_mut().downcast_mut::<LinearLayer>() {
                // Get number of weights and biases
                let weights_count = linear_layer.W.len() * linear_layer.W[0].len();
                let biases_count = linear_layer.b.len();

                // Unflatten weights
                for i in 0..linear_layer.W.len() {
                    for j in 0..linear_layer.W[i].len() {
                        linear_layer.W[i][j] = params[index];
                        index += 1;
                    }
                }

                // Unflatten biases
                for j in 0..linear_layer.b.len() {
                    linear_layer.b[j] = params[index];
                    index += 1;
                }
            }
            // Note: No need to unflatten ReLU layer as it has no parameters
        }
    }
}


fn generate_linear_dataset(
    num_samples: usize,
    num_features: usize,
    a: f64,
    b: f64,
    scale_factor: f64,
    noise_factor: f64,
    store: bool,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<Vec<f64>> = Vec::new();

    for _ in 0..num_samples {
        let mut input_row: Vec<f64> = Vec::new();
        let mut target_value = 0.0;

        for _ in 0..num_features {
            // Generate random input features between -100.0 and 100.0
            let input = rng.gen_range(-100.0..100.0);
            let input = (input * scale_factor * 100.0).round() / 100.0; // Round to 2 decimal places
            input_row.push(input);
            target_value += a * input;
        }

        // Apply the linear function and add noise
        target_value = target_value + b + (rng.gen_range(-noise_factor..noise_factor) * 100.0).round() / 100.0; // Round noise

        // Scale the target value if necessary
        targets.push(vec![(target_value * scale_factor * 100.0).round() / 100.0]); // Round target

        inputs.push(input_row);
    }

    (inputs, targets)
}


fn store_dataset(inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, hash: &Vec<u8>, signature: &Signature, path: String) -> io::Result<()> {
    let hash_encoded = hex::encode(hash); // Encode the hash to a base64 string

    let dataset = Dataset {
        inputs: inputs.clone(),
        targets: targets.clone(),
        hash: hash_encoded,
        signature: base64::encode(signature.to_bytes()),
    };

    let json = serde_json::to_string(&dataset)?;
    let mut file = File::create(path)?; // Create a file to store the dataset
    file.write_all(json.as_bytes())?;
    
    Ok(())
}

fn read_dataset( path: String) -> io::Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, String, String)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let dataset: Dataset = serde_json::from_reader(reader)?;


    Ok((dataset.inputs, dataset.targets, dataset.hash, dataset.signature))
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
fn hash_dataset(inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) -> Vec<u8> {
    // Create a Sha256 hasher
    let mut hasher = Sha256::new();

    // Serialize inputs
    for input_row in inputs {
        for &value in input_row {
            hasher.update(value.to_string());
            //println!("hash for:{}", value.to_string());

        }
    }

    // Serialize targets
    for target_row in targets {
        for &value in target_row {
            hasher.update(value.to_string());
            //println!("hash for:{}", value.to_string());

        }
    }
    //println!("hash complete");

    // Return the raw byte representation of the hash
    let end_hash = hasher.finalize().to_vec();
    println!("hash complete: {}",  hex::encode(end_hash.clone()));


    // Return the raw byte representation of the hash
    end_hash //hasher.finalize().to_vec()
}
// Function to serialize a Vec<f64> to a file
fn serialize_vector(file_path: &str, data: &Vec<f64>) -> io::Result<()> {
    let mut file = File::create(file_path)?;
    for &value in data {
        writeln!(file, "{}", value)?;
    }
    Ok(())
}

// Function to deserialize a Vec<f64> from a file
fn deserialize_vector(file_path: &str) -> io::Result<Vec<f64>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if let Ok(value) = line.trim().parse::<f64>() {
            data.push(value);
        }
    }
    Ok(data)
}
// Function to compute the hash of the model
fn hash_model(model: &Vec<f64>) -> Vec<u8> {
    let mut hasher = Sha256::new();
    for &value in model {
        // Round the value to 2 decimal places
        let rounded_value = (value * 100.0).round() / 100.0;

        // Hash the rounded value as bytes
        hasher.update(&rounded_value.to_le_bytes()); // Use the byte representation directly

        //println!("Hashing value: {}", rounded_value); // Print the rounded value for debugging
    }
    hasher.finalize().to_vec()
}


// Function to store model metadata
fn store_model_metadata(model: &Vec<f64>, hash: &Vec<u8>, signature: &Signature, path: String) -> io::Result<()> {
    let hash_encoded = hex::encode(hash); // Encode the hash to a base64 string

    // Create the model metadata
    let model_metadata = ModelMetadata {
        model: model.clone(),
        model_hash: hash_encoded,
        signature: base64::encode(signature.to_bytes()),
    };

    // Serialize and write the model metadata to a JSON file
    let json_metadata = serde_json::to_string(&model_metadata)?;
    //let mut file = File::create("model_metadata.json")?;
    let mut file = File::create(path)?;
    file.write_all(json_metadata.as_bytes())?;

    Ok(())
}// Function to store model metadata
fn store_global_model_metadata(model: &Vec<f64>, hash: &Vec<u8>, signature: &Signature, path: String) -> io::Result<()> {
    let hash_encoded = hex::encode(hash); // Encode the hash to a base64 string

    // Create the model metadata
    let model_metadata = ModelMetadata {
        model: model.clone(),
        model_hash: hash_encoded,
        signature: base64::encode(signature.to_bytes()),
    };

    // Serialize and write the model metadata to a JSON file
    let json_metadata = serde_json::to_string(&model_metadata)?;
    let mut file = File::create(path)?;
    //let mut file = File::create("global_model_metadata.json")?;
    file.write_all(json_metadata.as_bytes())?;

    Ok(())
}
fn read_model_metadata(path: String) -> io::Result<(Vec<f64>, String, String)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let model_metadata: ModelMetadata = serde_json::from_reader(reader)?;

  

    Ok((model_metadata.model, model_metadata.model_hash, model_metadata.signature))
}

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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Flag to generate a new dataset
    #[arg(short, long, default_value_t = false)]
    generate_dataset: bool,

    /// Type of previous training (1: tee, 2: zkvm)
    #[arg(short, long, default_value_t = 1)]
    previous_training_type: usize,

    /// Type of current computation (1: tee, 2: zkvm)
    #[arg(short, long, default_value_t = 1)]
    current_compute_type: usize,

    /// Flag to aggregate results
    #[arg(short, long, default_value_t = false)]
    aggregate: bool,

    /// Flag to indicate if training is active
    #[arg(short, long, default_value_t = false)]
    training: bool,

    /// Flag to create signing keys
    #[arg(short = 'k', long, default_value_t = false)]
    create_keys: bool,

    /// Number of aggregations to perform
    #[arg(short, long, default_value_t = 100)]
    num_aggregations: usize,

    /// Number of samples to read from the dataset
    #[arg(short , long, default_value_t = 80)]
    read_ds_num_samples: usize,

    /// Number of features to read from the dataset
    #[arg(short= 'f', long, default_value_t = 1)]
    read_ds_num_features: usize,

    /// Number of features to read from the dataset
    #[arg(short= 'g', long, default_value_t = 1)]
    type_global_model: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let args = Args::parse();

    // Use the parsed variables
    let bool_generate_dataset = args.generate_dataset;
    let type_previous_trainning = args.previous_training_type;
    let type_current_compute = args.current_compute_type;
    let bool_aggregate = args.aggregate;
    let bool_trainning = args.training;
    let create_keys = args.create_keys;
    let num_aggregations = args.num_aggregations;
    let read_ds_num_samples = args.read_ds_num_samples;
    let read_ds_num_features = args.read_ds_num_features;
    let bool_use_global_model = true;
    let type_global_model = args.type_global_model;; //(1: tee, 2: zkvm)


    if create_keys{
        println!("Creating keys for signing and verification");

        let signing_key = SigningKey::random(&mut OsRng); // Serialize with `::to_bytes()`
        let verifying_key = signing_key.verifying_key();
        store_keys(&signing_key, "./keys/signing_key.bin")?;
        store_verifying_key(&verifying_key, "./keys/verifying_key.bin")?;
        println!("Keys created and stored at ./keys/");
    }

    if bool_generate_dataset == true{
        
        let a = 1.0;
        let b = 0.0;
        let scale_factor = 1.0; 
        let noise_factor = 0.00001;
        for num_features in 1..=4 {
            // Loop for samples growing exponentially
            for exponent in 14..=14 { // 20, 40, 80, 160, 320, 640, 1280, 2560, ..., 200000
                let num_samples = 20 * 2u32.pow(exponent); // Exponential growth
    

                let (inputs, targets) = generate_linear_dataset(num_samples.try_into().unwrap(), num_features, a, b, scale_factor, noise_factor, bool_generate_dataset);

                let hash_ds = hash_dataset(&inputs,&targets);

                let read_signing_key = read_signing_key("./keys/signing_key.bin")?;
                let signing_key = read_signing_key;
                let signature: Signature = signing_key.sign(&hash_ds);

                let filename = format!("./datasets/dataset_samples_{}_features_{}.json", num_samples, num_features);

                let _ = store_dataset(&inputs, &targets, &hash_ds, &signature, filename);
            }
        }
    }

    if type_current_compute == 1 {
        if bool_trainning == true {
            let  mut global_model: Vec<f64> = Vec::new();
            if bool_use_global_model == true {

                if type_global_model == 1 {
                    println!("Verifying TEE Global Model");   

                   
                   
                    let filename_model = format!("./models/global_models/dataset_samples_20_features_1_aggs_2.json");                   

                    let (model, model_hash, signature) = read_model_metadata(filename_model)?;

                    let hash_model = hash_model(&model);
                    let verifying_key = read_verifying_key("./keys/verifying_key.bin")?;
                    let read_signature = Signature::from_slice(&base64::decode(signature).expect("Invalid base64 signature"))
                        .expect("Invalid signature");

                    let signature_valid = verify_signature(&verifying_key, &hex::decode(model_hash.clone())?, &read_signature);

                    let hash_valid = hex::encode(hash_model.clone()) == model_hash;
                   
                    println!("Hash is valid: {}", hash_valid);
                    println!("Signature is valid: {}", signature_valid);
                    println!("Verification complete");
                    global_model = model;
                }
                else if type_global_model == 2 {
                    println!("Verifying ZKVM Global Model");   

                    
                    let elf_path = "./elf_binaries/aggregation_from_tee"; // target_dir.join(&pkg_name).join(&target.name);
                    //let elf_path = "./elf_binaries/aggregation_from_zkvm"; // target_dir.join(&pkg_name).join(&target.name);
                    let elf = std::fs::read(&elf_path)
                        .with_context(|| format!("Failed to read ELF file at path: {}", elf_path))?;
                    let image_id = risc0_binfmt::compute_image_id(&elf)?;


                    //let receipt_path = format!("./verifiable_artifacts/receipts/receipts_aggregate_in_zkvm_with_zkvm_models_1.json");
                    let receipt_path = format!("./verifiable_artifacts/receipts/receipts_aggregate_in_zkvm_with_tee_models_1.json");
                   
                    let receipt: Receipt = bincode::deserialize(&fs::read(receipt_path)?)?;
                    
                    receipt.verify(image_id).expect(
                                "Code you have proven should successfully verify; did you specify the correct image ID?",
                            );
                    println!("Aggregation Proof Verified");

                    let (flattened_params, hash_validity, is_valid): (Vec<f64>, bool, bool) = receipt.journal.decode().unwrap();

                    println!("hash_validity: {}", hash_validity);
                    println!("signature_validity: {}", is_valid);
                    global_model = flattened_params;
                }
                else{

                }
            }

            let filename = format!("./datasets/dataset_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features);

            let (inputs, targets, read_hash, signature) = read_dataset(filename.to_string())?;

            let hash_ds = hash_dataset(&inputs,&targets);

            let hash_ds2 = hash_dataset(&inputs,&targets);

            let verifying_key = read_verifying_key("./keys/verifying_key.bin")?;
            let read_signature = Signature::from_slice(&base64::decode(signature).expect("Invalid base64 signature"))
                .expect("Invalid signature");

            let signature_valid =verify_signature(&verifying_key, &hex::decode(read_hash.clone())?, &read_signature);
            let hash_valid = hex::encode(hash_ds.clone()) == read_hash;
            let hash_valid2 = hex::encode(hash_ds.clone()) == hex::encode(hash_ds2.clone());
            

            println!("Hash is valid: {}", hash_valid);

            println!("Signature is valid: {}", signature_valid);
            println!("Verification complete");


            let mut nn = NN::new(inputs[0].len(), 2, targets[0].len()); // Use first row length for features

            if global_model.len()>0{
                nn.unflatten(&global_model);
            }
            // Training loop
            let learning_rate = 0.001; // Learning rate
            //for _ in 0..n_size {
            nn.train(&inputs, &targets, learning_rate);

            println!("233");
            let flattened_params = nn.layers.iter().flat_map(|layer| layer.flatten()).collect::<Vec<f64>>();

            let hash_model = hash_model(&flattened_params);
            let read_signing_key = read_signing_key("./keys/signing_key.bin")?;
            let signing_key = read_signing_key;
            let signature_model: Signature = signing_key.sign(&hash_model);

            let filename_model = format!("./models/multiple_models/dataset_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features);

            let _ = store_model_metadata(&flattened_params, &hash_model, &signature_model, filename_model);
            println!("Model stored and signed");
        




        } 

        if bool_aggregate == true {


            let mut models: Vec<Vec<f64>> = Vec::new();
            if type_previous_trainning == 1 {

                

                for _ in 0..num_aggregations {

                    let filename_model = format!("./models/multiple_models/dataset_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features);                   

                    let (model, model_hash, signature) = read_model_metadata(filename_model)?;

                    let hash_model = hash_model(&model);
                    let verifying_key = read_verifying_key("./keys/verifying_key.bin")?;
                    let read_signature = Signature::from_slice(&base64::decode(signature).expect("Invalid base64 signature"))
                        .expect("Invalid signature");

                    let signature_valid = verify_signature(&verifying_key, &hex::decode(model_hash.clone())?, &read_signature);

                    let hash_valid = hex::encode(hash_model.clone()) == model_hash;
                    
                    println!("Hash is valid: {}", model_hash);
                    println!("Hash is valid: {}", hex::encode(hash_model.clone()));
                    println!("Hash is valid: {}", hash_valid);
                    println!("Signature is valid: {}", signature_valid);
                    println!("Verification complete");
                    models.push(model);
                }

                
            }
            else if type_previous_trainning == 2 {
                for i_agg in 0..num_aggregations {

                    println!("Verifying ZKVM");   
                    
                    let elf_path = "./elf_binaries/trainning"; // target_dir.join(&pkg_name).join(&target.name);
                    let elf = std::fs::read(&elf_path)
                        .with_context(|| format!("Failed to read ELF file at path: {}", elf_path))?;
                    let image_id = risc0_binfmt::compute_image_id(&elf)?;

    
                    let receipt_path = format!("./verifiable_artifacts/receipts/receipts_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features);
                    let receipt: Receipt = bincode::deserialize(&fs::read(receipt_path)?)?;
                    
                    receipt.verify(image_id).expect(
                                "Code you have proven should successfully verify; did you specify the correct image ID?",
                            );
                    println!("Training Proof Verified {}", i_agg);

                    let (flattened_params, hash_validity, is_valid): (Vec<f64>, bool, bool) = receipt.journal.decode().unwrap();

                    println!("hash_validity: {}", hash_validity);
                    println!("signature_validity: {}", is_valid);
                    models.push(flattened_params);
                        
                }

            }
        
            else {
                println!("Type of previous training not supported (yet)");
            }

            if models.len() >  0 {
            let aggregated_model = average_model(&models);

            let hash_global_model = hash_model(&aggregated_model);
            let read_signing_key = read_signing_key("./keys/signing_key.bin")?;
            let signing_key = read_signing_key;
            let signature_model: Signature = signing_key.sign(&hash_global_model);
            let filename_global_model = format!("./models/global_models/dataset_samples_{}_features_{}_aggs_{}.json", read_ds_num_samples, read_ds_num_features, num_aggregations);
            let _ = store_global_model_metadata(&aggregated_model, &hash_global_model, &signature_model, filename_global_model);
            println!("Global Model stored and signed");
            }
            else {
                println!("No(t enough) models to aggregate");
            }
        }

    }
    else if type_current_compute == 2 {
        if bool_trainning == true {

            if bool_use_global_model == true {

                if type_global_model == 1 {

                    println!("Using global model from tee as base");
                    
                    let filename = format!("./datasets/dataset_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features);

                    let (inputs, targets, read_hash, signature) = read_dataset(filename)?;

                    let verifying_key = read_verifying_key("./keys/verifying_key.bin")?;
                    let read_signature = Signature::from_slice(&base64::decode(signature).expect("Invalid base64 signature"))
                        .expect("Invalid signature");

                                        
                    let gm_filename = format!("./models/global_models/dataset_samples_80_features_1.json");

                    let (global_model,  gm_read_hash, gm_signature) = read_model_metadata(gm_filename)?;

                    let gm_verifying_key = read_verifying_key("./keys/verifying_key.bin")?;
                    let gm_read_signature = Signature::from_slice(&base64::decode(gm_signature).expect("Invalid base64 signature"))
                        .expect("Invalid signature");


                    let elf_path = "./elf_binaries/trainning_from_tee"; // target_dir.join(&pkg_name).join(&target.name);
                    let elf = std::fs::read(&elf_path)
                    .with_context(|| format!("Failed to read ELF file at path: {}", elf_path))?;
                
            
                    let env = ExecutorEnv::builder()
                    .write(&inputs)
                    .expect("data failed to serialize")
                    .write(&targets)
                    .expect("data failed to serialize")
                    .write(&verifying_key.to_encoded_point(true))
                    .expect("data failed to serialize")
                    .write(&hex::decode(read_hash.clone())?)
                    .expect("data failed to serialize")
                    .write(&read_signature)
                    .expect("data failed to serialize")
                    .write(&gm_verifying_key.to_encoded_point(true))
                    .expect("data failed to serialize")
                    .write(&hex::decode(gm_read_hash.clone())?)
                    .expect("data failed to serialize")
                    .write(&gm_read_signature)
                    .expect("data failed to serialize")
                    .write(&global_model)
                    .expect("data failed to serialize")
                    .build()
                    .unwrap();
                    let prover = default_prover();
                    
                    let receipt = prover.prove(env, &elf).unwrap().receipt;
                    //let journ: risc0_zkvm::Journal = receipt.journal.bytes;

                    let receipt_path = format!("./verifiable_artifacts/receipts/receipts_learning_in_zkvm_with_tee_model_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features);
                    
                    fs::write(&receipt_path, bincode::serialize(&receipt).unwrap())?;
                    println!("Proof Receipt Stored in: {}",receipt_path);
                }
                else if type_global_model == 2 {

                    println!("Using global model from zkvm as base");
                    
                    let filename = format!("./datasets/dataset_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features);

                    let (inputs, targets, read_hash, signature) = read_dataset(filename)?;

                    let verifying_key = read_verifying_key("./keys/verifying_key.bin")?;
                    let read_signature = Signature::from_slice(&base64::decode(signature).expect("Invalid base64 signature"))
                        .expect("Invalid signature");
                    
                    let gm_elf_path: &str = "./elf_binaries/aggregation_from_zkvm"; // target_dir.join(&pkg_name).join(&target.name);
                    
                    let gm_elf = std::fs::read(&gm_elf_path)
                        .with_context(|| format!("Failed to read ELF file at path: {}", gm_elf_path))?;
                    let image_id = risc0_binfmt::compute_image_id(&gm_elf)?;

                    let receipt_path = format!("./verifiable_artifacts/receipts/receipts_aggregate_in_zkvm_with_zkvm_models_1.json");
                    //let receipt_path = format!("./models/global_models/dataset_samples_20_features_1_aggs_2.json");
                    let gm_receipt: Receipt = bincode::deserialize(&fs::read(receipt_path)?)?;
                    let gm_journal: Journal = gm_receipt.journal.clone();

                    let auxs: u8 = 1;
    
    
                    let elf_path_composed = "./elf_binaries/trainning_from_zkvm"; // target_dir.join(&pkg_name).join(&target.name);
                    let elf_compose = std::fs::read(&elf_path_composed)
                    .with_context(|| format!("Failed to read ELF file at path: {}", elf_path_composed))?;

                    println!("Building env");
        
    
    
                    let env = ExecutorEnv::builder()
                    .add_assumption(gm_receipt.clone())
                    .write(&inputs)
                    .expect("data failed to serialize")
                    .write(&targets)
                    .expect("data failed to serialize")
                    .write(&verifying_key.to_encoded_point(true))
                    .expect("data failed to serialize")
                    .write(&hex::decode(read_hash.clone())?)
                    .expect("data failed to serialize")
                    .write(&read_signature)
                    .expect("data failed to serialize")
                    .write(&gm_journal)
                    .expect("data failed to serialize")
                    .write(&image_id)
                    .expect("data failed to serialize")
                    .build().unwrap();

                    println!("Building default_prover");

                    let gm_receipt = default_prover()
                        .prove(env, &elf_compose)
                        .unwrap()
                        .receipt;


                    println!("NN Learning in ZKVM from another zkvm");
    
    
                    let gm_receipt_path = format!("./verifiable_artifacts/receipts/receipts_learning_in_zkvm_with_zkvm_model_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features);
                    
                    fs::write(&gm_receipt_path, bincode::serialize(&gm_receipt).unwrap())?;
                    println!("Proof Receipt Stored in: {}",gm_receipt_path);
                    
                }
                else {
                    println!("Base model from other VCC not supported (yet)");
                }

            }
            else {
                
                let filename = format!("./datasets/dataset_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features);

                let (inputs, targets, read_hash, signature) = read_dataset(filename)?;

                let verifying_key = read_verifying_key("./keys/verifying_key.bin")?;
                let read_signature = Signature::from_slice(&base64::decode(signature).expect("Invalid base64 signature"))
                    .expect("Invalid signature");

                let elf_path = "./elf_binaries/trainning"; // target_dir.join(&pkg_name).join(&target.name);
                let elf = std::fs::read(&elf_path)
                .with_context(|| format!("Failed to read ELF file at path: {}", elf_path))?;
            
        
                let env = ExecutorEnv::builder()
                .write(&inputs)
                .expect("data failed to serialize")
                .write(&targets)
                .expect("data failed to serialize")
                .write(&verifying_key.to_encoded_point(true))
                .expect("data failed to serialize")
                .write(&hex::decode(read_hash.clone())?)
                .expect("data failed to serialize")
                .write(&read_signature)
                .expect("data failed to serialize")
                    .build()
                    .unwrap();
                let prover = default_prover();
                
                let receipt = prover.prove(env, &elf).unwrap().receipt;
                //let journ: risc0_zkvm::Journal = receipt.journal.bytes;

                let receipt_path = format!("./verifiable_artifacts/receipts/receipts_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features);
                
                fs::write(&receipt_path, bincode::serialize(&receipt).unwrap())?;
                println!("Proof Receipt Stored in: {}",receipt_path);
            }
        }

        if bool_aggregate == true
        {
            
            
            if type_previous_trainning == 1 {

                let mut models: Vec<Vec<f64>> = Vec::new();
                let mut model_hashes: Vec<Vec<u8>> = Vec::new();
                let mut signatures: Vec<Signature> = Vec::new(); 
                let mut verifying_keys: Vec<EncodedPoint> = Vec::new(); 
                
                for _ in 0..num_aggregations {

                    let filename_model = format!("./models/multiple_models/dataset_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features); 

                    let (model, model_hash, signature) = read_model_metadata(filename_model)?;


                    let verifying_key: VerifyingKey = read_verifying_key("./keys/verifying_key.bin")?;
                    let read_signature: Signature = Signature::from_slice(&base64::decode(signature).expect("Invalid base64 signature"))
                        .expect("Invalid signature");

                    models.push(model);
                    model_hashes.push(hex::decode(model_hash.clone())?);
                    signatures.push(read_signature);
                    verifying_keys.push(verifying_key.to_encoded_point(true)); 
                }

                let elf_path = "./elf_binaries/aggregation_from_tee"; // target_dir.join(&pkg_name).join(&target.name);
                let elf = std::fs::read(&elf_path)
                .with_context(|| format!("Failed to read ELF file at path: {}", elf_path))?;
            
        
                let env = ExecutorEnv::builder()
                .write(&models)
                .expect("data failed to serialize")
                .write(&verifying_keys)
                .expect("data failed to serialize")
                .write(&model_hashes)
                .expect("data failed to serialize")
                .write(&signatures)
                .expect("data failed to serialize")
                .build()
                .unwrap();
                let prover = default_prover();
                
                let gm_receipt = prover.prove(env, &elf).unwrap().receipt;
                //let journ: risc0_zkvm::Journal = receipt.journal.bytes;
                println!(" Aggregated from TEE in zkvm");

                let receipt_path = format!("./verifiable_artifacts/receipts/receipts_aggregate_in_zkvm_with_tee_models_{}.json", num_aggregations);
                
                fs::write(&receipt_path, bincode::serialize(&gm_receipt).unwrap())?;
                println!("Proof Receipt Stored in: {}",receipt_path);




                
            }
            else if type_previous_trainning == 2 {
                let mut receipts: Vec<Receipt> = Vec::new();
                let mut journals: Vec<Journal> = Vec::new();
                let elf_path = "./elf_binaries/trainning"; // target_dir.join(&pkg_name).join(&target.name);
                let elf = std::fs::read(&elf_path)
                    .with_context(|| format!("Failed to read ELF file at path: {}", elf_path))?;
                let image_id = risc0_binfmt::compute_image_id(&elf)?;

                for i_agg in 0..num_aggregations {

                    println!("Verifying ZKVM");   
                    
                    
                    let receipt_path = format!("./verifiable_artifacts/receipts/receipts_samples_{}_features_{}.json", read_ds_num_samples, read_ds_num_features);
                    let receipt: Receipt = bincode::deserialize(&fs::read(receipt_path)?)?;
                    
                    journals.push(receipt.journal.clone());
                    receipts.push(receipt);

                }
                let auxs: u8 = 1;


                let elf_path_composed = "./elf_binaries/aggregation_from_zkvm"; // target_dir.join(&pkg_name).join(&target.name);
                let elf_compose = std::fs::read(&elf_path_composed)
                .with_context(|| format!("Failed to read ELF file at path: {}", elf_path_composed))?;
    


                let mut env = ExecutorEnv::builder();
                let mut env_aux = env.add_assumption(receipts[0].clone());
                for i_agg in 1..num_aggregations {
                env_aux = env_aux.add_assumption(receipts[i_agg].clone());
                }
                env_aux
                .write(&journals)
                .expect("data failed to serialize")
                .write(&image_id)
                .expect("data failed to serialize");
                let final_env = env_aux.build().unwrap();
    
                let gm_receipt = default_prover()
                    .prove(final_env, &elf_compose)
                    .unwrap()
                    .receipt;
                println!("NN Agregation and Receipts Verified inside ZKVM");


                let gm_receipt_path = format!("./verifiable_artifacts/receipts/receipts_aggregate_in_zkvm_with_zkvm_models_{}.json", num_aggregations);
                
                fs::write(&gm_receipt_path, bincode::serialize(&gm_receipt).unwrap())?;
                println!("Proof Receipt Stored in: {}",gm_receipt_path);

                //let image_id_compose = risc0_binfmt::compute_image_id(&elf_compose)?;
                //let _ = _receipt.verify(image_id_compose).expect(
                //"Code you have proven should successfully verify; did you specify the correct image ID?",
                //)
                
                
            }
            else {
                println!("Type of previous VCC not supported (yet)");
            }


        }

        
    }
    else {
        println!("Type of VCC not supported (yet)");
    }


    Ok(())
}
