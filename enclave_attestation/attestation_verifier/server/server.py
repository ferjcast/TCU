import socket
import json
import base64

from NsmUtil import NSMUtil
import threading
from queue import Queue
import time

# Global variables to store task state and result
task_queue = Queue(maxsize=1)
task_lock = threading.Lock()
current_result = None


def compute_task(number):
    """Simulated compute task that takes some time"""
    global current_result
    time.sleep(2)  # Simulate some computation time
    current_result =  b"Compute not ready yet {number * 2}"  # Simple computation: double the input
    
    task_queue.get()  # Remove task from queue
    task_queue.task_done()


def main():
    print("Starting server...")

    # Initialise NSMUtil
    nsm_util = NSMUtil()
    
    # Create a vsock socket object
    client_socket = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)

    # Listen for connection from any CID
    cid = socket.VMADDR_CID_ANY

    # The port should match the client running in parent EC2 instance
    client_port = 5000

    # Bind the socket to CID and port
    client_socket.bind((cid, client_port))

    # Listen for connection from client
    client_socket.listen()

    while True:
        client_connection, addr = client_socket.accept()

        # Get command from client
        payload = client_connection.recv(4096)
        request = json.loads(payload.decode())

        if request['action'] == 'info':
            print("action info")
            # Generate attestation document
            attestation_doc = nsm_util.get_attestation_doc()

            # Base64 encode the attestation doc
            attestation_doc_b64 = base64.b64encode(attestation_doc).decode()

            # Generate JSON request
            secretstore_request = json.dumps({
                'attestation_doc_b64': attestation_doc_b64
            })

            
            client_connection.sendall(str.encode(secretstore_request))

            # Close connection with secretstore

        # Close the connection with client

            
        elif request['action'] == 'compute':
            message = b"Computation will be performed"
            # Try to acquire lock and add to queue
            number = 2
            if task_lock.acquire(blocking=False):
                try:
                    task_queue.put(number)
                    # Start computation in a new thread
                    thread = threading.Thread(target=compute_task, args=(number,))
                    thread.start()
                    message = b'Computation started'
                finally:
                    task_lock.release()
            else:
                message = b'Server is busy with another task'
                
            print("message")
            print(message)
            
            signature = nsm_util.sign(message)
            
            message_request = json.dumps({
                
                'message': base64.b64encode(message).decode('utf-8'),
                'signature':base64.b64encode(signature).decode('utf-8')
            })
            print("message_request verify")
            print(nsm_util.verify_signature(message, signature))
            print("message_request")
            print(message_request)
            
            client_connection.sendall(str.encode(message_request))

            
            #print("Not defined info")
        elif request['action'] == 'proofRetrieval':
            #proof will be on the readfile .. check if file exists, otherwise return non-existant
            message = b"Compute ready"
            global current_result
            
            print("current_result")
            print(current_result)
            if current_result is None:
                print("Compute not ready")
                message = b'No results available yet'
            
            #message = current_result

            print("message")
            print(message)
            
            signature = nsm_util.sign(message)
            
            message_request = json.dumps({
                
                'message': base64.b64encode(message).decode('utf-8'),
                'signature':base64.b64encode(signature).decode('utf-8')
            })
            print("message_request verify")
            print(nsm_util.verify_signature(message, signature))
            print("message_request")
            print(message_request)
            
            client_connection.sendall(str.encode(message_request))

            
            print("Not defined info")
        else:
            print("Action not defined")

        client_connection.close()

if __name__ == '__main__':
    main()
