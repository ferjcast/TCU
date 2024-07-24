// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IVCCRegistry {
    function isRegisteredVCC(address vccId) external view returns (bool);
}

contract VCCTraceTrail {
    struct TraceEvent {
        bytes32 outputCommitment;     // C(Out_n)
        bytes32 internalInputCommitment; // C(IIn_n)
        bytes32 externalInputCommitment; // C(EIn_n) or C(Out_{n-1})
        address parentTraceEventVCC;
        bytes32 executionSpanId;
    }

    mapping(address => mapping(bytes32 => TraceEvent)) public traceEvents;
    IVCCRegistry public vccRegistry;

    event TraceEventLogged(
        address indexed vccId,
        bytes32 indexed eventId,
        bytes32 outputCommitment,
        bytes32 internalInputCommitment,
        bytes32 externalInputCommitment,
        address parentTraceEventVCC,
        bytes32 executionSpanId
    );

    constructor(address _vccRegistryAddress) {
        vccRegistry = IVCCRegistry(_vccRegistryAddress);
    }

    function logTraceEvent(
        bytes32 eventId,
        bytes32 outputCommitment,
        bytes32 internalInputCommitment,
        bytes32 externalInputCommitment,
        address parentTraceEventVCC,
        bytes32 executionSpanId
    ) external {
        require(vccRegistry.isRegisteredVCC(msg.sender), "Caller is not a registered VCC");
        require(traceEvents[msg.sender][eventId].outputCommitment == bytes32(0), "Trace event already exists");
        
        traceEvents[msg.sender][eventId] = TraceEvent(
            outputCommitment,
            internalInputCommitment,
            externalInputCommitment,
            parentTraceEventVCC,
            executionSpanId
        );
        
        emit TraceEventLogged(
            msg.sender,
            eventId,
            outputCommitment,
            internalInputCommitment,
            externalInputCommitment,
            parentTraceEventVCC,
            executionSpanId
        );
    }

    function getTraceEvent(address vccId, bytes32 eventId) external view returns (TraceEvent memory) {
        return traceEvents[vccId][eventId];
    }
}