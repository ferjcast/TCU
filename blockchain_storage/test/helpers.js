const { ethers } = require("hardhat");

async function measureGas(txPromise) {
  const tx = await txPromise;
  const receipt = await tx.wait();
  return receipt.gasUsed;
}

module.exports = {
  measureGas
};