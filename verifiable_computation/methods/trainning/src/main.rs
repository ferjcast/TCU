use risc0_zkvm::guest::env;
use std::any::Any;
use k256::{
    ecdsa::{signature::Verifier, Signature, VerifyingKey},
    EncodedPoint,
};

use k256::sha2::{Digest, Sha256};
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
                *w -= self.dL_dW[i][j].clamp(-1.0, 1.0) * learning_rate;
            }
            self.b[i] -= self.dL_db[i].clamp(-1.0, 1.0) * learning_rate;
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
    /*pub fn unflatten(&mut self, params: &Vec<f64>) {
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
    }*/
}
fn hash_dataset(inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) ->  Vec<u8> {
    // Create a Sha256 hasher
    let mut hasher = Sha256::new();

    // Serialize inputs
    for input_row in inputs {
        for &value in input_row {
            hasher.update(value.to_string());
        }
    }

    // Serialize targets
    for target_row in targets {
        for &value in target_row {
            hasher.update(value.to_string());
        }
    }

    // Return the hexadecimal representation of the hash
    let result = hasher.finalize();
    result.to_vec() //hex::encode(result)
}

fn main() {
    // Read input data and target data from the environment using RISC Zero ZKVM
    let inputs: Vec<Vec<f64>> = env::read(); // Adjust to read a batch of inputs
    let targets: Vec<Vec<f64>> = env::read(); // Adjust to read a batch of targets
    //let n_size: u32 = env::read(); // Number of rows in the dataset
    let encoded_verifying_key: EncodedPoint = env::read();
    let message: Vec<u8> = env::read();
    let signature: Signature = env::read();

    let verifying_key = VerifyingKey::from_encoded_point(&encoded_verifying_key).unwrap();
    let is_valid = verifying_key.verify(&message, &signature).is_ok();
    println!("is_valid {}", is_valid);
    let ds_hash: Vec<u8> = hash_dataset(&inputs, &targets);

    let mut hash_validity = false;

    if ds_hash == message {
        hash_validity = true;
        println!("random_matrix_hash and message are equal.");
    }

    // Create neural network (input features based on dataset, hidden neurons, output features)
    let mut nn = NN::new(inputs[0].len(), 2, targets[0].len()); // Use first row length for features

    // Training loop
    let learning_rate = 0.001; // Learning rate
    //for _ in 0..n_size {
    nn.train(&inputs, &targets, learning_rate);
    //}
    let flattened_params = nn.layers.iter().flat_map(|layer| layer.flatten()).collect::<Vec<f64>>();
/* 
    // Get predictions after training
    let predictions = nn.forward(&inputs);

    // Flatten the neural network parameters
    let flattened_params = nn.layers.iter().flat_map(|layer| layer.flatten()).collect::<Vec<f64>>();

    // Create a new neural network and unflatten the parameters
    let mut nn_reconstructed = NN::new(inputs[0].len(), 4, targets[0].len());
    nn_reconstructed.unflatten(&flattened_params);

    // Get predictions from the reconstructed neural network
    let reconstructed_predictions = nn_reconstructed.forward(&inputs);

    // Verify that the predictions are the same for both NNs
    assert_eq!(predictions, reconstructed_predictions, "Predictions do not match!");
    println!("Predictions did match");
*/
    // Commit the predictions to the RISC Zero journal
    env::commit(&(flattened_params, hash_validity, is_valid)); 
    println!(
        "Total cycles for guest code execution: {}",
        env::cycle_count()
    );
/* 
    let predictions = nn.forward(&inputs);
    // Log the total cycle count for the guest code execution
    println!(
        "Total cycles for guest code execution: {}",
        env::cycle_count()
    );
    for (i, (pred_row, target_row, input_row)) in predictions.iter()
    .zip(targets.iter())
    .zip(inputs.iter())
    .map(|((pred_row, target_row), input_row)| (pred_row, target_row, input_row))
    .enumerate() 
{
    // Divide predictions by 10,000
    let adjusted_preds: Vec<f64> = pred_row.iter().map(|&val| val / 1.0).collect();
    
    // Print the adjusted predictions alongside the original targets and inputs
    println!(
        "Row {}: Input: {:?}, Predicted: {:?}, Target: {:?}", 
        i, 
        input_row, 
        adjusted_preds, 
        target_row
    );
}*/
}

