import sys
import socket
import json
import base64
import cbor2
import cose
import base64
import time 

from cose import EC2, CoseAlgorithms, CoseEllipticCurves
from Crypto.Util.number import long_to_bytes
from OpenSSL import crypto

from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA

from Crypto.Hash import SHA256
from Crypto.Signature import pss

def verify_attestation_doc(attestation_doc, pcrs = [], root_cert_pem = None):
    """
    Verify the attestation document
    If invalid, raise an exception
    """
    # Decode CBOR attestation document
    data = cbor2.loads(attestation_doc)
    
    print("verify_attestation_doc")
    print("data")
    print(data)

    # Load and decode document payload
    doc = data[2]
    doc_obj = cbor2.loads(doc)
    
    print("doc_obj")
    print(doc_obj)

    print("public_key")
    print(doc_obj["public_key"])
    
    rsa_pkey = RSA.import_key(doc_obj["public_key"])
    print("rsa_pkey")
    print(rsa_pkey.exportKey())

    # Get PCRs from attestation document
    document_pcrs_arr = doc_obj['pcrs']

    ###########################
    # Part 1: Validating PCRs #
    ###########################
    for index, pcr in enumerate(pcrs):
        # Attestation document doesn't have specified PCR, raise exception
        if index not in document_pcrs_arr or document_pcrs_arr[index] is None:
            print("Wrong PCR%s" % index)

        # Get PCR hexcode
        doc_pcr = document_pcrs_arr[index].hex()

        # Check if PCR match
        if pcr != doc_pcr:
            print("Wrong PCR%s" % index)


    ################################
    # Part 2: Validating signature #
    ################################

    # Get signing certificate from attestation document
    cert = crypto.load_certificate(crypto.FILETYPE_ASN1, doc_obj['certificate'])

    # Get the key parameters from the cert public key
    cert_public_numbers = cert.get_pubkey().to_cryptography_key().public_numbers()
    x = cert_public_numbers.x
    y = cert_public_numbers.y
    curve = cert_public_numbers.curve

    x = long_to_bytes(x)
    y = long_to_bytes(y)

    # Create the EC2 key from public key parameters
    key = EC2(alg = CoseAlgorithms.ES384, x = x, y = y, crv = CoseEllipticCurves.P_384)

    # Get the protected header from attestation document
    phdr = cbor2.loads(data[0])

    # Construct the Sign1 message
    msg = cose.Sign1Message(phdr = phdr, uhdr = data[1], payload = doc)
    msg.signature = data[3]

    # Verify the signature using the EC2 key
    if not msg.verify_signature(key):
        raise Exception("Wrong signature")


    ##############################################
    # Part 3: Validating signing certificate PKI #
    ##############################################
    if root_cert_pem is not None:
        # Create an X509Store object for the CA bundles
        store = crypto.X509Store()

        # Create the CA cert object from PEM string, and store into X509Store
        _cert = crypto.load_certificate(crypto.FILETYPE_PEM, root_cert_pem)
        store.add_cert(_cert)

        # Get the CA bundle from attestation document and store into X509Store
        # Except the first certificate, which is the root certificate
        for _cert_binary in doc_obj['cabundle'][1:]:
            _cert = crypto.load_certificate(crypto.FILETYPE_ASN1, _cert_binary)
            store.add_cert(_cert)

        # Get the X509Store context
        store_ctx = crypto.X509StoreContext(store, cert)
        
        # Validate the certificate
        # If the cert is invalid, it will raise exception
        store_ctx.verify_certificate()
    return rsa_pkey
    

def verify_signature(message, signature, public_key):
        """
        Verify the signature for the given message using the RSA public key.

        :param message: The original message (as bytes).
        :param signature: The signature to verify (as bytes).
        :return: True if valid, False otherwise.
        """
        h = SHA256.new(message)
        verifier = pss.new(public_key)

        # If you prefer PKCS#1 v1.5, uncomment below and comment out the PSS lines:
        # verifier = pkcs1_15.new(self._rsa_key.publickey())

        try:
            verifier.verify(h, signature)
            return True
        except (ValueError, TypeError):
            return False
def main():
        # Get the root cert PEM content
    with open('root.pem', 'r') as file:
        root_cert_pem = file.read()
        
    # Create a vsock socket object
    s = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    
    # Get CID from command line parameter
    cid = int(sys.argv[1])
    
    pcr0 = sys.argv[2]
    print("pcr0")
    print(pcr0)
    # The port should match the server running in enclave
    port = 5000
    
    

    # Connect to the server
    s.connect((cid, port))

    # Send command to the server running in enclave
    s.send(str.encode(json.dumps({
        'action': 'info'
    })))

    # receive the plaintext from the server and print it to console
    response = s.recv(65536)
    
    request = json.loads(response.decode())

    # # Get attestation document
    attestation_doc_b64 = request['attestation_doc_b64']
    attestation_doc = base64.b64decode(attestation_doc_b64)
    
    rsa_pkey = verify_attestation_doc(attestation_doc, pcrs = [pcr0], root_cert_pem = root_cert_pem)

    #print(request)

    # close the connection 
    s.close()
    
    
    ###Verify signature
    s2 = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    
    # Get CID from command line parameter

    # The port should match the server running in enclave
    port = 5000
    
    

    # Connect to the server
    s2.connect((cid, port))
        # Send command to the server running in enclave
    s2.send(str.encode(json.dumps({
        'action': 'compute',
        'external_input': 'aa',
        'internal_input': 'aa2'
    })))

    # receive the plaintext from the server and print it to console
    response = s2.recv(65536)
    
    request = json.loads(response.decode())


    # # Get attestation document
    message_signed = base64.b64decode(request['message'])
    signature = base64.b64decode(request['signature'])
    print("verify_signature")
    print(verify_signature(message_signed, signature, rsa_pkey))

    print("request")
    print(request)
    print("message_signed")
    print(message_signed)
    

    # close the connection 
    s2.close()
    
        
    time.sleep(3)
    ###Verify signature
    s3 = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    
    # Get CID from command line parameter

    # The port should match the server running in enclave
    port = 5000
    
    

    # Connect to the server
    s3.connect((cid, port))
        # Send command to the server running in enclave
    s3.send(str.encode(json.dumps({
        'action': 'proofRetrieval'
    })))

    # receive the plaintext from the server and print it to console
    response = s3.recv(65536)
    
    request = json.loads(response.decode())


    # # Get attestation document
    message_signed = base64.b64decode(request['message'])
    signature = base64.b64decode(request['signature'])
    print("verify_signature")
    print(verify_signature(message_signed, signature, rsa_pkey))

    print("request")
    print(request)
    print("message_signed")
    print(message_signed)
    

    # close the connection 
    s3.close()

if __name__ == '__main__':
    main()
