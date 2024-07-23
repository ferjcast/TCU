# Verifiable Computation Experiments

This project implements and compares different approaches to verifiable computation using Rust. It includes implementations of matrix multiplication using Trusted Execution Environments (TEEs) and Zero-Knowledge Virtual Machines (zkVMs), along with performance measurements and comparisons.

## Features

- Matrix multiplication implementations:
  - Standard algorithm
  - Using TEEs (AWS Nitro Enclaves)
  - Using zkVMs (risc0)
- Performance measurements for various matrix sizes
- Comparison of proving and verification times for different approaches

## Prerequisites

- Rust (latest stable version)
- Cargo
- AWS account and configured credentials (for TEE experiments)
- risc0 toolchain (for zkVM experiments)

## Setup

1. Install Rust and Cargo:

```
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install risc0 (for zkVM experiments):

```
    curl -L https://risczero.com/install | bash
```

## Running Experiments

1. Build the project:
```
    cargo build --release
```
2. Build the ELFs (ZKVM Binaries):
```
    ./build_elfs.sh
```
3. Run the experiments:
```
    ./run-local.sh
```
This will run matrix multiplication experiments for various sizes using different approaches and record the results in CSV files.

4. Build EIF in nitro AWS-EC2

Note: Modify Dockerfile if you want to use already built rust binary to reduce image size

```
    ./build_enclave.sh
```
5. Run enclave in nitro AWS-EC2
```
    ./run_enclave.sh
```



## Results

After running the experiments, following files are created:

- `proving_tee_results.csv`: TEE proving measurements
- `proving_zkvm_results.csv`: zkVM proving measurements
- `verifying_tee_inside_tee_results.csv`: TEE verification measurements (inside TEE)
- `verifying_zkvm_inside_tee_results.csv`: zkVM verification measurements (inside TEE)
- `verifying_tee_inside_zkvm_results.csv`: TEE verification measurements (inside zkVM)
- `verifying_zkvm_inside_zkvm_results.csv`: zkVM verification measurements (inside zkVM)

- `verifying_tee_inside_tee_w_proving_results.csv`: TEE verification and proving measurements (inside TEE)
- `verifying_zkvm_inside_tee_w_proving_results.csv`: zkVM verification and proving measurements (inside TEE)
- `verifying_tee_inside_zkvm_w_proving_results.csv`: TEE verification and proving measurements (inside zkVM)
- `verifying_zkvm_inside_zkvm_w_proving_results.csv`: zkVM verification and proving measurements (inside zkVM)


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the APACHE License - see the LICENSE file for details.