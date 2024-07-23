 nitro-cli terminate-enclave --all

 #version with buiding binary inside the image
 #nitro-cli run-enclave --cpu-count 2 --memory 72828 --enclave-cid 16 --eif-path verification-app.eif --debug-mode && nitro-cli console --enclave-id $(nitro-cli describe-enclaves | jq -r '.[0].EnclaveID')

 #version with already built binary
 nitro-cli run-enclave --cpu-count 2 --memory 12138 --enclave-cid 16 --eif-path verification-app.eif --debug-mode && nitro-cli console --enclave-id $(nitro-cli describe-enclaves | jq -r '.[0].EnclaveID')