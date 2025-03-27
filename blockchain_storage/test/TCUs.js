const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("TCU System", () => {
  let TCURegistry;
  let TCUTraceTrail;
  let tcuRegistry;
  let tcuTraceTrail;
  let owner;
  let tcu1;
  let tcu2;
  let tcu3;

  beforeEach(async () => {
    [owner, tcu1, tcu2, tcu3] = await ethers.getSigners();

    TCURegistry = await ethers.getContractFactory("TCURegistry");
    tcuRegistry = await TCURegistry.deploy();

    TCUTraceTrail = await ethers.getContractFactory("TCUTraceTrail");
    tcuTraceTrail = await TCUTraceTrail.deploy(await tcuRegistry.getAddress());
  });

  describe("TCURegistry", () => {
    it("Should register a new TCU", async () => {
      const verificationKey = "ipfs://QmeeQhGoyMQc7eQWERE88kFFq4WbdVRrjHctZhH1hPHNds--VerificationKey1";
      const publicKey =       "ipfs://QmeeQhGoyMQc7eQWERE88kFFq4WbdVRrjHctZhH1hPHNds------QmPublicKey1";

      await expect(tcuRegistry.connect(tcu1).registerTCU(
        verificationKey,
        publicKey,
        ethers.ZeroAddress,
        tcu2.address,
        await tcuTraceTrail.getAddress()
      )).to.emit(tcuRegistry, "TCURegistered");

      const registeredTCU = await tcuRegistry.getTCU(tcu1.address);
      const [returnedVerificationKey, returnedPublicKey, predecessorId, successorId, traceTrailAddress] = registeredTCU;

      expect(returnedVerificationKey).to.equal(verificationKey);
      expect(returnedPublicKey).to.equal(publicKey);
      expect(predecessorId).to.equal(ethers.ZeroAddress);
      expect(successorId).to.equal(tcu2.address);
      expect(traceTrailAddress).to.equal(await tcuTraceTrail.getAddress());
    });

    it("Should not allow registering the same TCU twice", async () => {
        const verificationKey = "ipfs://QmeeQhGoyMQc7eQWERE88kFFq4WbdVRrjHctZhH1hPHNds--VerificationKey1";
        const publicKey =       "ipfs://QmeeQhGoyMQc7eQWERE88kFFq4WbdVRrjHctZhH1hPHNds------QmPublicKey1";
  

      await tcuRegistry.connect(tcu1).registerTCU(
        verificationKey,
        publicKey,
        ethers.ZeroAddress,
        tcu2.address,
        await tcuTraceTrail.getAddress()
      );

      await expect(
        tcuRegistry.connect(tcu1).registerTCU(
          verificationKey,
          publicKey,
          ethers.ZeroAddress,
          tcu2.address,
          await tcuTraceTrail.getAddress()
        )
      ).to.be.revertedWith("TCU already registered");
    });

    it("Should correctly identify registered TCUs", async () => {
        const verificationKey = "ipfs://QmeeQhGoyMQc7eQWERE88kFFq4WbdVRrjHctZhH1hPHNds--VerificationKey1";
        const publicKey =       "ipfs://QmeeQhGoyMQc7eQWERE88kFFq4WbdVRrjHctZhH1hPHNds------QmPublicKey1";
  
      await tcuRegistry.connect(tcu1).registerTCU(
        verificationKey,
        publicKey,
        ethers.ZeroAddress,
        tcu2.address,
        await tcuTraceTrail.getAddress()
      );

      expect(await tcuRegistry.isRegisteredTCU(tcu1.address)).to.be.true;
      expect(await tcuRegistry.isRegisteredTCU(tcu2.address)).to.be.false;
    });
  });

  describe("TCUTraceTrail", () => {
    beforeEach(async () => {
      // Register TCUs
      const verificationKey = "ipfs://QmeeQhGoyMQc7eQWERE88kFFq4WbdVRrjHctZhH1hPHNds--VerificationKey1";
      const publicKey =       "ipfs://QmeeQhGoyMQc7eQWERE88kFFq4WbdVRrjHctZhH1hPHNds------QmPublicKey1";


      await tcuRegistry.connect(tcu1).registerTCU(
        verificationKey,
        publicKey,
        ethers.ZeroAddress,
        tcu2.address,
        await tcuTraceTrail.getAddress()
      );

      await tcuRegistry.connect(tcu2).registerTCU(
        verificationKey,
        publicKey,
        tcu1.address,
        tcu3.address,
        await tcuTraceTrail.getAddress()
      );
    });

    it("Should log a trace event for a registered TCU", async () => {
      const eventId = ethers.encodeBytes32String("event1");
      const outputCommitment = ethers.encodeBytes32String("output1");
      const internalInputCommitment = ethers.encodeBytes32String("internal1");
      const externalInputCommitment = ethers.encodeBytes32String("external1");
      const executionSpanId = ethers.encodeBytes32String("span1");

      await expect(
        tcuTraceTrail.connect(tcu1).logTraceEvent(
          eventId,
          outputCommitment,
          internalInputCommitment,
          externalInputCommitment,
          ethers.ZeroAddress,
          executionSpanId
        )
      ).to.emit(tcuTraceTrail, "TraceEventLogged");

      const loggedEvent = await tcuTraceTrail.getTraceEvent(tcu1.address, eventId);
      expect(loggedEvent.outputCommitment).to.equal(outputCommitment);
      expect(loggedEvent.internalInputCommitment).to.equal(internalInputCommitment);
      expect(loggedEvent.externalInputCommitment).to.equal(externalInputCommitment);
    });

    it("Should not allow logging a trace event for an unregistered TCU", async () => {
      const eventId = ethers.encodeBytes32String("event1");
      const outputCommitment = ethers.encodeBytes32String("output1");
      const internalInputCommitment = ethers.encodeBytes32String("internal1");
      const externalInputCommitment = ethers.encodeBytes32String("external1");
      const executionSpanId = ethers.encodeBytes32String("span1");

      await expect(
        tcuTraceTrail.connect(tcu3).logTraceEvent(
          eventId,
          outputCommitment,
          internalInputCommitment,
          externalInputCommitment,
          tcu2.address,
          executionSpanId
        )
      ).to.be.revertedWith("Caller is not a registered TCU");
    });

    it("Should not allow logging the same trace event twice", async () => {
      const eventId = ethers.encodeBytes32String("event1");
      const outputCommitment = ethers.encodeBytes32String("output1");
      const internalInputCommitment = ethers.encodeBytes32String("internal1");
      const externalInputCommitment = ethers.encodeBytes32String("external1");
      const executionSpanId = ethers.encodeBytes32String("span1");

      await tcuTraceTrail.connect(tcu1).logTraceEvent(
        eventId,
        outputCommitment,
        internalInputCommitment,
        externalInputCommitment,
        ethers.ZeroAddress,
        executionSpanId
      );

      await expect(
        tcuTraceTrail.connect(tcu1).logTraceEvent(
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