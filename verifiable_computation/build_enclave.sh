#!/bin/bash

rm verification-app.eif
cargo clean
docker build -t verification-app .
nitro-cli build-enclave --docker-uri verification-app --output-file verification-app.eif