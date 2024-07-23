cargo risczero build --manifest-path ./methods/matrix_multiplication_proving/Cargo.toml
cargo risczero build --manifest-path ./methods/matrix_multiplication_tee_verification/Cargo.toml
cargo risczero build --manifest-path ./methods/matrix_multiplication_zkvm_verification/Cargo.toml

cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/matrix_multiplication_proving/matrix_multiplication_proving elf_binaries/
cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/matrix_multiplication_tee_verification/matrix_multiplication_tee_verification elf_binaries/
cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/matrix_multiplication_zkvm_verification/matrix_multiplication_zkvm_verification elf_binaries/