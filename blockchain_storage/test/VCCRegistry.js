const { expect } = require("chai");
const { ethers } = require("hardhat");
const { measureGas } = require("./helpers");

describe("VCCRegistry", function () {
  let VCCRegistry;
  let vccRegistry;

  beforeEach(async function () {
    VCCRegistry = await ethers.getContractFactory("VCCRegistry");
    vccRegistry = await VCCRegistry.deploy();
  });

  it("Should measure contract size", async function () {
    const bytecode = VCCRegistry.bytecode;
    console.log(`VCCRegistry contract size: ${bytecode.length / 2} bytes`);
  });

  it("Should register a new VCC and measure gas", async function () {
    const id = ethers.keccak256(ethers.toUtf8Bytes("vcc1"));
    const vk = ethers.keccak256(ethers.toUtf8Bytes("verificationKey"));
    const pk = ethers.keccak256(ethers.toUtf8Bytes("publicKey"));
    const predecessorId = ethers.keccak256(ethers.toUtf8Bytes("predecessor"));
    const successorId = ethers.keccak256(ethers.toUtf8Bytes("successor"));
    const auditTrailAddress = ethers.Wallet.createRandom().address;
  
    const gasUsed = await measureGas(
      vccRegistry.registerVCC(id, vk, pk, predecessorId, successorId, auditTrailAddress)
    );
    console.log(`Gas used for registerVCC: ${gasUsed}`);
  
    const registeredVCC = await vccRegistry.getVCC(id);
    expect(registeredVCC.verificationKey).to.equal(vk);
    expect(registeredVCC.publicKey).to.equal(pk);
    expect(registeredVCC.predecessorId).to.equal(predecessorId);
    expect(registeredVCC.successorId).to.equal(successorId);
    expect(registeredVCC.auditTrailAddress).to.equal(auditTrailAddress);
  });
  it("Should emit VCCRegistered event", async function () {
    const id = ethers.keccak256(ethers.toUtf8Bytes("vcc2"));
    const vk = ethers.keccak256(ethers.toUtf8Bytes("verificationKey"));
    const pk = ethers.keccak256(ethers.toUtf8Bytes("publicKey"));
    const predecessorId = ethers.keccak256(ethers.toUtf8Bytes("predecessor"));
    const successorId = ethers.keccak256(ethers.toUtf8Bytes("successor"));
    const auditTrailAddress = ethers.Wallet.createRandom().address;
  
    await expect(vccRegistry.registerVCC(id, vk, pk, predecessorId, successorId, auditTrailAddress))
      .to.emit(vccRegistry, "VCCRegistered")
      .withArgs(id, vk, pk, predecessorId, successorId, auditTrailAddress);
  });

  it("Should correctly return VCC data", async function () {
    const id = ethers.keccak256(ethers.toUtf8Bytes("vcc1"));
    const vk = ethers.keccak256(ethers.toUtf8Bytes("verificationKey"));
    const pk = ethers.keccak256(ethers.toUtf8Bytes("publicKey"));
    const predecessorId = ethers.keccak256(ethers.toUtf8Bytes("predecessor"));
    const successorId = ethers.keccak256(ethers.toUtf8Bytes("successor"));
    const auditTrailAddress = ethers.Wallet.createRandom().address;
  
    await vccRegistry.registerVCC(id, vk, pk, predecessorId, successorId, auditTrailAddress);
  
    const retrievedVCC = await vccRegistry.getVCC(id);
    expect(retrievedVCC.verificationKey).to.equal(vk);
    expect(retrievedVCC.publicKey).to.equal(pk);
    expect(retrievedVCC.predecessorId).to.equal(predecessorId);
    expect(retrievedVCC.successorId).to.equal(successorId);
    expect(retrievedVCC.auditTrailAddress).to.equal(auditTrailAddress);
  });

  it("Should not allow registering the same VCC twice", async function () {
    const id = ethers.keccak256(ethers.toUtf8Bytes("vcc1"));
    const vk = ethers.keccak256(ethers.toUtf8Bytes("verificationKey"));
    const pk = ethers.keccak256(ethers.toUtf8Bytes("publicKey"));
    const predecessorId = ethers.keccak256(ethers.toUtf8Bytes("predecessor"));
    const successorId = ethers.keccak256(ethers.toUtf8Bytes("successor"));
    const auditTrailAddress = ethers.Wallet.createRandom().address;

    await vccRegistry.registerVCC(id, vk, pk, predecessorId, successorId, auditTrailAddress);

    await expect(vccRegistry.registerVCC(id, vk, pk, predecessorId, successorId, auditTrailAddress))
      .to.be.revertedWith("VCC already registered");
  });
});