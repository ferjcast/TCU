// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TCURegistry {
    struct TCU {
        string verificationKeyURI;
        string publicKeyURI;
        address predecessorId;
        address successorId;
        address traceTrailAddress;
    }

    mapping(address => TCU) public tcus;

    event TCURegistered(
        address indexed id,
        string verificationKeyURI,
        string publicKeyURI,
        address predecessorId,
        address successorId,
        address traceTrailAddress
    );

    function registerTCU(
        string memory verificationKeyURI,
        string memory publicKeyURI,
        address predecessorId,
        address successorId,
        address traceTrailAddress
    ) external {
        require(bytes(tcus[msg.sender].verificationKeyURI).length == 0, "TCU already registered");
        
        tcus[msg.sender] = TCU(verificationKeyURI, publicKeyURI, predecessorId, successorId, traceTrailAddress);
        
        emit TCURegistered(msg.sender, verificationKeyURI, publicKeyURI, predecessorId, successorId, traceTrailAddress);
    }

    function getTCU(address id) external view returns (TCU memory) {
        return tcus[id];
    }

    function isRegisteredTCU(address tcuId) external view returns (bool) {
        return bytes(tcus[tcuId].verificationKeyURI).length > 0;
    }
}