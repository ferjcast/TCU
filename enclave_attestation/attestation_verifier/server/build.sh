#!/bin/bash

rm attestation-verifier-app.eif
#docker rmi -f $(docker images -a -q)

docker build -t attestation-verifier-app .

#nitro-cli build-enclave --docker-dir ./ --docker-uri richardfan1126/nitro-enclave-python-demo:latest --output-file attestation-verifier-app.eif
nitro-cli build-enclave --docker-uri attestation-verifier-app --output-file attestation-verifier-app.eif

# nitro-cli run-enclave --cpu-count 2 --memory 2048 --enclave-cid 16 --eif-path attestation-verifier-app.eif --debug-mode && nitro-cli console --enclave-id $(nitro-cli describe-enclaves | jq -r '.[0].EnclaveID')

# sudo systemctl restart nitro-enclaves-allocator.service
# sudo systemctl status nitro-enclaves-allocator.service
# nitro-cli terminate-enclave --all

# 5267465953856ec545fed07a45a482dc90633015171e2cc378d6d9bee3441bb4c4c9c52aef712bb25a3c9420267fae81

#last      "PCR0": "8b7cda85406cebcd6f501b89e72903e0e9f61fc06d0382d97427c8d8d9f8d7a19fef22012fce5e82a5a3bac30813f6e7",
#python3 secretstore.py 7f94b0de01d4e4650aae6d2ff0532cfb598371ad545243f23e0a7133d40eee366d246b72419bb2406f2c7e713458e719
#python client.py 16