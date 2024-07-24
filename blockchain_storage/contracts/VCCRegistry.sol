// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VCCRegistry {
    struct VCC {
        string verificationKeyURI;
        string publicKeyURI;
        address predecessorId;
        address successorId;
        address traceTrailAddress;
    }

    mapping(address => VCC) public vccs;

    event VCCRegistered(
        address indexed id,
        string verificationKeyURI,
        string publicKeyURI,
        address predecessorId,
        address successorId,
        address traceTrailAddress
    );

    function registerVCC(
        string memory verificationKeyURI,
        string memory publicKeyURI,
        address predecessorId,
        address successorId,
        address traceTrailAddress
    ) external {
        require(bytes(vccs[msg.sender].verificationKeyURI).length == 0, "VCC already registered");
        
        vccs[msg.sender] = VCC(verificationKeyURI, publicKeyURI, predecessorId, successorId, traceTrailAddress);
        
        emit VCCRegistered(msg.sender, verificationKeyURI, publicKeyURI, predecessorId, successorId, traceTrailAddress);
    }

    function getVCC(address id) external view returns (VCC memory) {
        return vccs[id];
    }

    function isRegisteredVCC(address vccId) external view returns (bool) {
        return bytes(vccs[vccId].verificationKeyURI).length > 0;
    }
}