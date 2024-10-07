# VeriFlow Smart Contracts

This project contains the smart contracts for the VeriFlow system, implemented using Solidity and tested with Hardhat.

## Overview

The project includes two main contracts:

1. **VCCRegistry**: Manages the registration of Verifiable Composable Components (VCCs).
2. **AuditTrail**: Handles the logging of trace events for workflow execution.

## Prerequisites

- Node.js (v18.16.1 or later)
- npm (v9.5.1 or later)

## Setup

Install dependencies:

```shell
npm install
```

## Running Tests

To run the tests, use the following command:

```shell
npx hardhat test
```

This will run all tests and display the results, including contract sizes and gas costs for operations.
