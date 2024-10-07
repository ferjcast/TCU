cargo risczero build --manifest-path ./methods/matrix_multiplication_proving/Cargo.toml
cargo risczero build --manifest-path ./methods/matrix_multiplication_tee_verification/Cargo.toml
cargo risczero build --manifest-path ./methods/matrix_multiplication_zkvm_verification/Cargo.toml
cargo risczero build --manifest-path ./methods/matrix_multiplication_tee_verification_w_proving/Cargo.toml
cargo risczero build --manifest-path ./methods/matrix_multiplication_zkvm_verification_w_proving/Cargo.toml


cargo risczero build --manifest-path ./methods/aggregation_from_tee/Cargo.toml
cargo risczero build --manifest-path ./methods/aggregation_from_zkvm/Cargo.toml
cargo risczero build --manifest-path ./methods/trainning_from_tee/Cargo.toml
cargo risczero build --manifest-path ./methods/trainning_from_zkvm/Cargo.toml


cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/matrix_multiplication_proving/matrix_multiplication_proving elf_binaries/
cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/matrix_multiplication_tee_verification/matrix_multiplication_tee_verification elf_binaries/
cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/matrix_multiplication_zkvm_verification/matrix_multiplication_zkvm_verification elf_binaries/
cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/matrix_multiplication_tee_verification_w_proving/matrix_multiplication_tee_verification_w_proving elf_binaries/
cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/matrix_multiplication_zkvm_verification_w_proving/matrix_multiplication_zkvm_verification_w_proving elf_binaries/

cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/aggregation_from_tee/aggregation_from_tee elf_binaries/
cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/aggregation_from_zkvm/aggregation_from_zkvm elf_binaries/
cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/trainning_from_tee/trainning_from_tee elf_binaries/
cp target/riscv-guest/riscv32im-risc0-zkvm-elf/docker/trainning_from_zkvm/trainning_from_zkvm elf_binaries/