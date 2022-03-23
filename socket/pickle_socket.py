import socket, pickle
from struct import pack, unpack
from enum import Enum

class FLAGS(Enum):
  SETUP = 0
  START_TRAIN = 1
  HEALTH_CODE = 2
  TERMINATE = 3

  RESULT_OK = 3
  RESULT_BAD = 4

import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

class Message(object):
  def __init__(self, source, flag, data = None):
    self.source = source
    self.flag = flag 
    # Server -> Client 
    #        0 : compile config (optim, loss, metrics) - dict
    #        1 : model architecture - dict

    # Client -> Server
    #        2 : model parameters - list(np.array) 
    #        3 : status code - int STATUS_CODE
    #        4 : data idxs - {int client id: list(int)}
    self.data = data

  def __len__(self):
    return int(bool(self.data))

  def __sizeof__(self):
    return get_size(self.data) / 1000000 # MB
  

class Server(object):
  clients = {}
  def __init__(self, host, port, max_con):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.bind((host, port))
    self.socket.listen(max_con)

    
  def __del__(self):
    for id, client in self.clients.items():
      client["client"].close()

  def accept(self, id):
    if id in self.clients:
      self.close(id) # close connection to the client with the id 

    client, client_addr = self.socket.accept()
    self.clients[id] = {}
    self.clients[id]["client"] = client
    self.clients[id]["addr"] = client_addr

  def send(self, id, data):
    if id not in self.clients:
      self.accept(id)

    _send(self.clients[id]['client'], data)
  
  def recv(self, id):
    if id not in self.clients == None:
      raise Exception('Cannot receive data, no client is connected')
    
    return _recv(self.clients[id]['client'])

  def close(self, id):
    self.clients[id]["client"].close()
    del self.clients[id]

  def close_socket(self):
    if self.socket:
      self.socket.close()
      self.socket = None

class Client(object):
  socket = None
  id = None
  def __del__(self):
    self.close()

  def connect(self, id, host, port):
    self.id = id 
    self.socket = socket.socket()
    self.socket.connect((host, port))
    print(f"client {id} connection succeeded")
    return self

  def send(self, data):
    if not self.socket:
      raise Exception('You have to connect first before sending data')
    _send(self.socket, data)
    return self

  def recv(self):
    if not self.socket:
      raise Exception('You have to connect first before receiving data')
    return _recv(self.socket)

  def recv_and_close(self):
    data = self.recv()
    self.close()
    return data

  def close(self):
    if self.socket:
      self.socket.close()
      self.socket = None

### TODO: Fix below to use buffer size 1024 to recv data

## helper functions ##
def _send(socket, data):
  data = pickle.dumps(data, protocol=4)
  data = pack('>I', len(data)) + data
  socket.sendall(data)

def _recv(socket):
  raw_msglen = recvall(socket, 4)
  #raw_msglen = recvall(socket, 4)
  if not raw_msglen:
      return None
  msglen = unpack('>I', raw_msglen)[0]
  msg =  recvall(socket, msglen)
  return pickle.loads(msg)
  
def recvall(socket, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    cursor = 0
    buffer_size = 1024
    while len(data) < n:
        #packet = socket.recv(n - len(data))
        #if n - len(data) > buffer_size:
        #  packet = socket.recv(buffer_size)
        #else:
        #  packet = socket.recv(n - len(data))
        
        packet = socket.recv(1)
        if not packet:
            break
        data.extend(packet)
    return data