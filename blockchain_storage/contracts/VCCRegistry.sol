// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VCCRegistry {
    struct VCC {
        bytes32 verificationKey;
        bytes32 publicKey;
        bytes32 predecessorId;
        bytes32 successorId;
        address auditTrailAddress;
    }

    mapping(bytes32 => VCC) public vccs;

    event VCCRegistered(bytes32 indexed id, bytes32 verificationKey, bytes32 publicKey, bytes32 predecessorId, bytes32 successorId, address auditTrailAddress);

    function registerVCC(
        bytes32 id,
        bytes32 verificationKey,
        bytes32 publicKey,
        bytes32 predecessorId,
        bytes32 successorId,
        address auditTrailAddress
    ) external {
        require(vccs[id].verificationKey == bytes32(0), "VCC already registered");
        
        vccs[id] = VCC(verificationKey, publicKey, predecessorId, successorId, auditTrailAddress);
        
        emit VCCRegistered(id, verificationKey, publicKey, predecessorId, successorId, auditTrailAddress);
    }

    function getVCC(bytes32 id) external view returns (VCC memory) {
        return vccs[id];
    }
}