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
    vccTraceTrail = await VCCTraceTrail.deploy(vccRegistry.target);
  });

  describe("VCCRegistry", () => {
    it("Should register a new VCC", async () => {
      const verificationKeyURI = "ipfs://QmVerificationKey1";
      const publicKeyURI = "ipfs://QmPublicKey1";

      await expect(vccRegistry.connect(vcc1).registerVCC(
        verificationKeyURI,
        publicKeyURI,
        ethers.ZeroAddress,
        vcc2.address,
        vccTraceTrail.target
      )).to.emit(vccRegistry, "VCCRegistered");

      const registeredVCC = await vccRegistry.getVCC(vcc1.address);
      expect(registeredVCC.verificationKeyURI).to.equal(verificationKeyURI);
      expect(registeredVCC.publicKeyURI).to.equal(publicKeyURI);
    });

    it("Should not allow registering the same VCC twice", async () => {
      const verificationKeyURI = "ipfs://QmVerificationKey1";
      const publicKeyURI = "ipfs://QmPublicKey1";

      await vccRegistry.connect(vcc1).registerVCC(
        verificationKeyURI,
        publicKeyURI,
        ethers.ZeroAddress,
        vcc2.address,
        vccTraceTrail.target
      );

      await expect(
        vccRegistry.connect(vcc1).registerVCC(
          verificationKeyURI,
          publicKeyURI,
          ethers.ZeroAddress,
          vcc2.address,
          vccTraceTrail.target
        )
      ).to.be.revertedWith("VCC already registered");
    });

    it("Should correctly identify registered VCCs", async () => {
      const verificationKeyURI = "ipfs://QmVerificationKey1";
      const publicKeyURI = "ipfs://QmPublicKey1";

      await vccRegistry.connect(vcc1).registerVCC(
        verificationKeyURI,
        publicKeyURI,
        ethers.ZeroAddress,
        vcc2.address,
        vccTraceTrail.target
      );

      expect(await vccRegistry.isRegisteredVCC(vcc1.address)).to.be.true;
      expect(await vccRegistry.isRegisteredVCC(vcc2.address)).to.be.false;
    });
  });

  describe("VCCTraceTrail", () => {
    beforeEach(async () => {
      // Register VCCs
      const verificationKeyURI = "ipfs://QmVerificationKey";
      const publicKeyURI = "ipfs://QmPublicKey";

      await vccRegistry.connect(vcc1).registerVCC(
        verificationKeyURI,
        publicKeyURI,
        ethers.ZeroAddress,
        vcc2.address,
        vccTraceTrail.target
      );

      await vccRegistry.connect(vcc2).registerVCC(
        verificationKeyURI,
        publicKeyURI,
        vcc1.address,
        vcc3.address,
        vccTraceTrail.target
      );
    });

    it("Should log a trace event for a registered VCC", async () => {
      const eventIdURI = "ipfs://QmEvent1";
      const outputCommitmentURI = "ipfs://QmOutput1";
      const internalInputCommitmentURI = "ipfs://QmInternal1";
      const externalInputCommitmentURI = "ipfs://QmExternal1";
      const executionSpanIdURI = "ipfs://QmSpan1";

      await expect(
        vccTraceTrail.connect(vcc1).logTraceEvent(
          eventIdURI,
          outputCommitmentURI,
          internalInputCommitmentURI,
          externalInputCommitmentURI,
          ethers.ZeroAddress,
          executionSpanIdURI
        )
      ).to.emit(vccTraceTrail, "TraceEventLogged");

      const loggedEvent = await vccTraceTrail.getTraceEvent(vcc1.address, eventIdURI);
      expect(loggedEvent.outputCommitmentURI).to.equal(outputCommitmentURI);
      expect(loggedEvent.internalInputCommitmentURI).to.equal(internalInputCommitmentURI);
      expect(loggedEvent.externalInputCommitmentURI).to.equal(externalInputCommitmentURI);
    });

    it("Should not allow logging a trace event for an unregistered VCC", async () => {
      const eventIdURI = "ipfs://QmEvent1";
      const outputCommitmentURI = "ipfs://QmOutput1";
      const internalInputCommitmentURI = "ipfs://QmInternal1";
      const externalInputCommitmentURI = "ipfs://QmExternal1";
      const executionSpanIdURI = "ipfs://QmSpan1";

      await expect(
        vccTraceTrail.connect(vcc3).logTraceEvent(
          eventIdURI,
          outputCommitmentURI,
          internalInputCommitmentURI,
          externalInputCommitmentURI,
          vcc2.address,
          executionSpanIdURI
        )
      ).to.be.revertedWith("Caller is not a registered VCC");
    });

    it("Should not allow logging the same trace event twice", async () => {
      const eventIdURI = "ipfs://QmEvent1";
      const outputCommitmentURI = "ipfs://QmOutput1";
      const internalInputCommitmentURI = "ipfs://QmInternal1";
      const externalInputCommitmentURI = "ipfs://QmExternal1";
      const executionSpanIdURI = "ipfs://QmSpan1";

      await vccTraceTrail.connect(vcc1).logTraceEvent(
        eventIdURI,
        outputCommitmentURI,
        internalInputCommitmentURI,
        externalInputCommitmentURI,
        ethers.ZeroAddress,
        executionSpanIdURI
      );

      await expect(
        vccTraceTrail.connect(vcc1).logTraceEvent(
          eventIdURI,
          outputCommitmentURI,
          internalInputCommitmentURI,
          externalInputCommitmentURI,
          ethers.ZeroAddress,
          executionSpanIdURI
        )
      ).to.be.revertedWith("Trace event already exists");
    });
  });
});