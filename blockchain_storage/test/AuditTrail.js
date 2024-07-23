const { expect } = require("chai");
const { ethers } = require("hardhat");
const { measureGas } = require("./helpers");

describe("AuditTrail", function () {
  let AuditTrail;
  let auditTrail;

  beforeEach(async function () {
    AuditTrail = await ethers.getContractFactory("AuditTrail");
    auditTrail = await AuditTrail.deploy();
  });

  it("Should measure contract size", async function () {
    const bytecode = AuditTrail.bytecode;
    console.log(`AuditTrail contract size: ${bytecode.length / 2} bytes`);
  });

  it("Should log a new trace event and measure gas", async function () {
    const id = ethers.keccak256(ethers.toUtf8Bytes("traceEvent1"));
    const externalInputHash = ethers.keccak256(ethers.toUtf8Bytes("externalInput"));
    const internalInputHash = ethers.keccak256(ethers.toUtf8Bytes("internalInput"));
    const parentTraceEventId = ethers.keccak256(ethers.toUtf8Bytes("parentTraceEvent"));
    const executionSpanId = ethers.keccak256(ethers.toUtf8Bytes("executionSpan"));
  
    const gasUsed = await measureGas(
      auditTrail.logTraceEvent(id, externalInputHash, internalInputHash, parentTraceEventId, executionSpanId)
    );
    console.log(`Gas used for logTraceEvent: ${gasUsed}`);
  
    const loggedTraceEvent = await auditTrail.getTraceEvent(id);
    expect(loggedTraceEvent.externalInputHash).to.equal(externalInputHash);
    expect(loggedTraceEvent.internalInputHash).to.equal(internalInputHash);
    expect(loggedTraceEvent.parentTraceEventId).to.equal(parentTraceEventId);
    expect(loggedTraceEvent.executionSpanId).to.equal(executionSpanId);
  });
  
  it("Should emit TraceEventLogged event", async function () {
    const id = ethers.keccak256(ethers.toUtf8Bytes("traceEvent2"));
    const externalInputHash = ethers.keccak256(ethers.toUtf8Bytes("externalInput"));
    const internalInputHash = ethers.keccak256(ethers.toUtf8Bytes("internalInput"));
    const parentTraceEventId = ethers.keccak256(ethers.toUtf8Bytes("parentTraceEvent"));
    const executionSpanId = ethers.keccak256(ethers.toUtf8Bytes("executionSpan"));
  
    await expect(auditTrail.logTraceEvent(id, externalInputHash, internalInputHash, parentTraceEventId, executionSpanId))
      .to.emit(auditTrail, "TraceEventLogged")
      .withArgs(id, externalInputHash, internalInputHash, parentTraceEventId, executionSpanId);
  });

  it("Should correctly return TraceEvent data", async function () {
    const id = ethers.keccak256(ethers.toUtf8Bytes("traceEvent1"));
    const externalInputHash = ethers.keccak256(ethers.toUtf8Bytes("externalInput"));
    const internalInputHash = ethers.keccak256(ethers.toUtf8Bytes("internalInput"));
    const parentTraceEventId = ethers.keccak256(ethers.toUtf8Bytes("parentTraceEvent"));
    const executionSpanId = ethers.keccak256(ethers.toUtf8Bytes("executionSpan"));
  
    await auditTrail.logTraceEvent(id, externalInputHash, internalInputHash, parentTraceEventId, executionSpanId);
  
    const retrievedTraceEvent = await auditTrail.getTraceEvent(id);
    expect(retrievedTraceEvent.externalInputHash).to.equal(externalInputHash);
    expect(retrievedTraceEvent.internalInputHash).to.equal(internalInputHash);
    expect(retrievedTraceEvent.parentTraceEventId).to.equal(parentTraceEventId);
    expect(retrievedTraceEvent.executionSpanId).to.equal(executionSpanId);
  });

  it("Should not allow logging the same trace event twice", async function () {
    const id = ethers.keccak256(ethers.toUtf8Bytes("traceEvent1"));
    const externalInputHash = ethers.keccak256(ethers.toUtf8Bytes("externalInput"));
    const internalInputHash = ethers.keccak256(ethers.toUtf8Bytes("internalInput"));
    const parentTraceEventId = ethers.keccak256(ethers.toUtf8Bytes("parentTraceEvent"));
    const executionSpanId = ethers.keccak256(ethers.toUtf8Bytes("executionSpan"));

    await auditTrail.logTraceEvent(id, externalInputHash, internalInputHash, parentTraceEventId, executionSpanId);

    await expect(auditTrail.logTraceEvent(id, externalInputHash, internalInputHash, parentTraceEventId, executionSpanId))
      .to.be.revertedWith("Trace event already exists");
  });
});