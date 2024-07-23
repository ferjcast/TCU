# Final stage: Create the runtime image
FROM ubuntu:20.04

# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /usr/src/app

# Copy the built binary from the builder stage

# Install necessary build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc-9 \
    g++-9 \
    curl \
    wget \
    pkg-config \
    libssl-dev \
    tzdata \
    time \
    git \
    python3 \
    bc

COPY ./scripts/install_rust.sh ./scripts/install_rust.sh
COPY ./scripts/install_risc0.sh ./scripts/install_risc0.sh


ENV TERM=xterm



#RUN chmod +x measurements_tee.sh
#INSTALL RUST
#RUN chmod +x scripts/install_rust.sh scripts/install_risc0.sh
RUN ./scripts/install_rust.sh
#RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Source the environment file
SHELL ["/bin/bash", "-c"]
RUN source $HOME/.cargo/env
ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_HOME="/root/.rustup"
ENV CARGO_HOME="/root/.cargo"
#INSTALL RISC0 INSTALLER
RUN ./scripts/install_risc0.sh
#RUN curl -L https://risczero.com/install | bash
#INSTALL RISC0 BIN
ENV PATH="$PATH:/root/.risc0/bin"


##RISC0 and BUILD can be skipped using a local built, so the docker image and eif are smaller
RUN rzup --verbose --version 1.0.1

COPY . .

RUN cd host
RUN cargo build --release
RUN cd ..
RUN mv target/release/host host_final
RUN chmod +x measurements_tee.sh

CMD ["/usr/src/app/measurements_tee.sh", "/usr/src/app/host_final"]
#CMD ["bash"]
