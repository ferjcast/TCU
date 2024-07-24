const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("VCC System", () => {
  let VCCRegistry;
  let VCCTraceTrail;
  let vccRegistry;
  let vccTraceTrail;
  let owner;
  let vcc1;
  let vcc2;
  let vcc3;

  beforeEach(async () => {
    [owner, vcc1, vcc2, vcc3] = await ethers.getSigners();

    VCCRegistry = await ethers.getContractFactory("VCCRegistry");
    vccRegistry = await VCCRegistry.deploy();

    VCCTraceTrail = await ethers.getContractFactory("VCCTraceTrail");
    vccTraceTrail = await VCCTraceTrail.deploy(await vccRegistry.getAddress());
  });

  describe("VCCRegistry", () => {
    it("Should register a new VCC", async () => {
      const verificationKey = ethers.encodeBytes32String("vk1");
      const publicKey = ethers.encodeBytes32String("pk1");

      await expect(vccRegistry.connect(vcc1).registerVCC(
        verificationKey,
        publicKey,
        ethers.ZeroAddress,
        vcc2.address,
        await vccTraceTrail.getAddress()
      )).to.emit(vccRegistry, "VCCRegistered");

      const registeredVCC = await vccRegistry.getVCC(vcc1.address);
      const [returnedVerificationKey, returnedPublicKey, predecessorId, successorId, traceTrailAddress] = registeredVCC;

      expect(returnedVerificationKey).to.equal(verificationKey);
      expect(returnedPublicKey).to.equal(publicKey);
      expect(predecessorId).to.equal(ethers.ZeroAddress);
      expect(successorId).to.equal(vcc2.address);
      expect(traceTrailAddress).to.equal(await vccTraceTrail.getAddress());
    });

    it("Should not allow registering the same VCC twice", async () => {
      const verificationKey = ethers.encodeBytes32String("vk1");
      const publicKey = ethers.encodeBytes32String("pk1");

      await vccRegistry.connect(vcc1).registerVCC(
        verificationKey,
        publicKey,
        ethers.ZeroAddress,
        vcc2.address,
        await vccTraceTrail.getAddress()
      );

      await expect(
        vccRegistry.connect(vcc1).registerVCC(
          verificationKey,
          publicKey,
          ethers.ZeroAddress,
          vcc2.address,
          await vccTraceTrail.getAddress()
        )
      ).to.be.revertedWith("VCC already registered");
    });

    it("Should correctly identify registered VCCs", async () => {
      const verificationKey = ethers.encodeBytes32String("vk1");
      const publicKey = ethers.encodeBytes32String("pk1");

      await vccRegistry.connect(vcc1).registerVCC(
        verificationKey,
        publicKey,
        ethers.ZeroAddress,
        vcc2.address,
        await vccTraceTrail.getAddress()
      );

      expect(await vccRegistry.isRegisteredVCC(vcc1.address)).to.be.true;
      expect(await vccRegistry.isRegisteredVCC(vcc2.address)).to.be.false;
    });
  });

  describe("VCCTraceTrail", () => {
    beforeEach(async () => {
      // Register VCCs
      const verificationKey = ethers.encodeBytes32String("vk");
      const publicKey = ethers.encodeBytes32String("pk");

      await vccRegistry.connect(vcc1).registerVCC(
        verificationKey,
        publicKey,
        ethers.ZeroAddress,
        vcc2.address,
        await vccTraceTrail.getAddress()
      );

      await vccRegistry.connect(vcc2).registerVCC(
        verificationKey,
        publicKey,
        vcc1.address,
        vcc3.address,
        await vccTraceTrail.getAddress()
      );
    });

    it("Should log a trace event for a registered VCC", async () => {
      const eventId = ethers.encodeBytes32String("event1");
      const outputCommitment = ethers.encodeBytes32String("output1");
      const internalInputCommitment = ethers.encodeBytes32String("internal1");
      const externalInputCommitment = ethers.encodeBytes32String("external1");
      const executionSpanId = ethers.encodeBytes32String("span1");

      await expect(
        vccTraceTrail.connect(vcc1).logTraceEvent(
          eventId,
          outputCommitment,
          internalInputCommitment,
          externalInputCommitment,
          ethers.ZeroAddress,
          executionSpanId
        )
      ).to.emit(vccTraceTrail, "TraceEventLogged");

      const loggedEvent = await vccTraceTrail.getTraceEvent(vcc1.address, eventId);
      expect(loggedEvent.outputCommitment).to.equal(outputCommitment);
      expect(loggedEvent.internalInputCommitment).to.equal(internalInputCommitment);
      expect(loggedEvent.externalInputCommitment).to.equal(externalInputCommitment);
    });

    it("Should not allow logging a trace event for an unregistered VCC", async () => {
      const eventId = ethers.encodeBytes32String("event1");
      const outputCommitment = ethers.encodeBytes32String("output1");
      const internalInputCommitment = ethers.encodeBytes32String("internal1");
      const externalInputCommitment = ethers.encodeBytes32String("external1");
      const executionSpanId = ethers.encodeBytes32String("span1");

      await expect(
        vccTraceTrail.connect(vcc3).logTraceEvent(
          eventId,
          outputCommitment,
          internalInputCommitment,
          externalInputCommitment,
          vcc2.address,
          executionSpanId
        )
      ).to.be.revertedWith("Caller is not a registered VCC");
    });

    it("Should not allow logging the same trace event twice", async () => {
      const eventId = ethers.encodeBytes32String("event1");
      const outputCommitment = ethers.encodeBytes32String("output1");
      const internalInputCommitment = ethers.encodeBytes32String("internal1");
      const externalInputCommitment = ethers.encodeBytes32String("external1");
      const executionSpanId = ethers.encodeBytes32String("span1");

      await vccTraceTrail.connect(vcc1).logTraceEvent(
        eventId,
        outputCommitment,
        internalInputCommitment,
        externalInputCommitment,
        ethers.ZeroAddress,
        executionSpanId
      );

      await expect(
        vccTraceTrail.connect(vcc1).logTraceEvent(
          eventId,
          outputCommitment,
          internalInputCommitment,
          externalInputCommitment,
          ethers.ZeroAddress,
          executionSpanId
        )
      ).to.be.revertedWith("Trace event already exists");
    });
  });
});