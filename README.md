# Trusted Compute Units

This repository contains the implementation of Trusted Compute Units, a system for enabling secure, verifiable, and auditable cross-organizational workflows.

## Project Structure

The project is divided into three main components:

1. `blockchain_storage/`: Implementation of the Decentralized Management Platform (DMP)
2. `enclave_attestation/`: Tools and utilities for Trusted Execution Environment (TEE) attestation
3. `verifiable_computation/`: Implementation of Computing Part (AL) using TEEs and zkVMs

## Getting Started

Test each component by following the instructions in their respective READMEs:
- [Blockchain Storage Setup](./blockchain_storage/README.md)
- [Enclave Attestation Setup](./enclave_attestation/README.md)
- [Verifiable Computation Setup](./verifiable_computation/README.md)

## Components

### Blockchain Storage

The `blockchain_storage/` directory contains the smart contract implementation for the Decentralized Management Platform. This includes:

- TCU Registry
- Audit Trail
- Smart contracts for managing workflow states

To set up and run the blockchain component, refer to the [**README**](./blockchain_storage/README.md) in the `blockchain_storage/` directory.

### Enclave Attestation

The `enclave_attestation/` directory contains tools and utilities for working with Trusted Execution Environments (TEEs), specifically AWS Nitro Enclaves. This includes:

- Attestation verification tools
- Secure communication utilities

For more information on setting up and using the enclave attestation tools, see the [**README**](./enclave_attestation/README.md) in the `enclave_attestation/` directory.

### Verifiable Computation

The `verifiable_computation/` directory contains the implementation of Verifiable Computation Component (VCC) using both TEEs and zkVMs. This includes:

- Federating Learning Training and Aggregation implementations
- Matrix multiplication implementations
- Proving and verification modules
- Performance measurement tools

To run experiments and benchmarks, refer to the  [**README**](./verifiable_computation/README.md) in the `verifiable_computation/` directory.




## Contributing

Contributions to this project are welcome. Please feel free to submit issues and pull requests.

## License

Apache License, Version 2.0

## Contact

For any questions or concerns, please open an issue in this repository.