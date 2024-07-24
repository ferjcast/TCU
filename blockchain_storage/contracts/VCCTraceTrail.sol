// SPDX-License-Identifier: MIT
import "./VCCRegistry.sol";

contract VCCTraceTrail {
    struct TraceEvent {
        string outputCommitmentURI;
        string internalInputCommitmentURI;
        string externalInputCommitmentURI;
        address parentTraceEventVCC;
        string executionSpanIdURI;
    }

    mapping(address => mapping(string => TraceEvent)) public traceEvents;
    VCCRegistry public vccRegistry;

    event TraceEventLogged(
        address indexed vccId,
        string indexed eventIdURI,
        string outputCommitmentURI,
        string internalInputCommitmentURI,
        string externalInputCommitmentURI,
        address parentTraceEventVCC,
        string executionSpanIdURI
    );

    constructor(address _vccRegistryAddress) {
        vccRegistry = VCCRegistry(_vccRegistryAddress);
    }

    function logTraceEvent(
        string memory eventIdURI,
        string memory outputCommitmentURI,
        string memory internalInputCommitmentURI,
        string memory externalInputCommitmentURI,
        address parentTraceEventVCC,
        string memory executionSpanIdURI
    ) external {
        require(vccRegistry.isRegisteredVCC(msg.sender), "Caller is not a registered VCC");
        require(bytes(traceEvents[msg.sender][eventIdURI].outputCommitmentURI).length == 0, "Trace event already exists");
        
        traceEvents[msg.sender][eventIdURI] = TraceEvent(
            outputCommitmentURI,
            internalInputCommitmentURI,
            externalInputCommitmentURI,
            parentTraceEventVCC,
            executionSpanIdURI
        );
        
        emit TraceEventLogged(
            msg.sender,
            eventIdURI,
            outputCommitmentURI,
            internalInputCommitmentURI,
            externalInputCommitmentURI,
            parentTraceEventVCC,
            executionSpanIdURI
        );
    }

    function getTraceEvent(address vccId, string memory eventIdURI) external view returns (TraceEvent memory) {
        return traceEvents[vccId][eventIdURI];
    }
}