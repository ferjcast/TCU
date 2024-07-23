// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AuditTrail {
    struct TraceEvent {
        bytes32 externalInputHash;
        bytes32 internalInputHash;
        bytes32 parentTraceEventId;
        bytes32 executionSpanId;
    }

    mapping(bytes32 => TraceEvent) public traceEvents;

    event TraceEventLogged(bytes32 indexed id, bytes32 externalInputHash, bytes32 internalInputHash, bytes32 parentTraceEventId, bytes32 executionSpanId);

    function logTraceEvent(
        bytes32 id,
        bytes32 externalInputHash,
        bytes32 internalInputHash,
        bytes32 parentTraceEventId,
        bytes32 executionSpanId
    ) external {
        require(traceEvents[id].externalInputHash == bytes32(0), "Trace event already exists");
        
        traceEvents[id] = TraceEvent(externalInputHash, internalInputHash, parentTraceEventId, executionSpanId);
        
        emit TraceEventLogged(id, externalInputHash, internalInputHash, parentTraceEventId, executionSpanId);
    }

    function getTraceEvent(bytes32 id) external view returns (TraceEvent memory) {
        return traceEvents[id];
    }
}