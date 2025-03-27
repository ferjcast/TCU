// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface ITCURegistry {
    function isRegisteredTCU(address tcuId) external view returns (bool);
}

contract TCUTraceTrail {
    struct TraceEvent {
        bytes32 outputCommitment;     // C(Out_n)
        bytes32 internalInputCommitment; // C(IIn_n)
        bytes32 externalInputCommitment; // C(EIn_n) or C(Out_{n-1})
        address parentTraceEventTCU;
        bytes32 executionSpanId;
    }

    mapping(address => mapping(bytes32 => TraceEvent)) public traceEvents;
    ITCURegistry public tcuRegistry;

    event TraceEventLogged(
        address indexed tcuId,
        bytes32 indexed eventId,
        bytes32 outputCommitment,
        bytes32 internalInputCommitment,
        bytes32 externalInputCommitment,
        address parentTraceEventTCU,
        bytes32 executionSpanId
    );

    constructor(address _tcuRegistryAddress) {
        tcuRegistry = ITCURegistry(_tcuRegistryAddress);
    }

    function logTraceEvent(
        bytes32 eventId,
        bytes32 outputCommitment,
        bytes32 internalInputCommitment,
        bytes32 externalInputCommitment,
        address parentTraceEventTCU,
        bytes32 executionSpanId
    ) external {
        require(tcuRegistry.isRegisteredTCU(msg.sender), "Caller is not a registered TCU");
        require(traceEvents[msg.sender][eventId].outputCommitment == bytes32(0), "Trace event already exists");
        
        traceEvents[msg.sender][eventId] = TraceEvent(
            outputCommitment,
            internalInputCommitment,
            externalInputCommitment,
            parentTraceEventTCU,
            executionSpanId
        );
        
        emit TraceEventLogged(
            msg.sender,
            eventId,
            outputCommitment,
            internalInputCommitment,
            externalInputCommitment,
            parentTraceEventTCU,
            executionSpanId
        );
    }

    function getTraceEvent(address tcuId, bytes32 eventId) external view returns (TraceEvent memory) {
        return traceEvents[tcuId][eventId];
    }
}