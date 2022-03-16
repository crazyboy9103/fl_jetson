import socket, pickle
from struct import pack, unpack
from enum import Enum


class CLASSIFY_MODELS(Enum):
  RESNET18 = 0
  VGG16 = 1
  MOBILENETV2 = 2

class DETECT_MODELS(Enum):
  SSDLITE = 3
  SSD = 4

class DATASET(Enum):
  MNIST = 0
  CIFAR10 = 1
  CIFAR100 = 2
  COCO = 3
  
class FLAGS(Enum):
  SETUP = 0
  HEALTH_CODE = 1
  START_TRAIN = 2
  TERMINATE = 3

  RESULT_OK = 4
  RESULT_BAD = 5

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
  def __init__(self, source, flag, data=None):
    self.source = source  # message source id : -1 for server or non-negative integer 
    self.flag = flag  # See FLAGS enum 
    self.data = data  # Data to be pickled

  def __len__(self):
    return int(self.data != None)

  def __sizeof__(self):
    return get_size(self.data) / 1000000 # MB
  
  def get_data(self):
    if self.flag == FLAGS.SETUP:
      return self.data["dataset_name"], self.data["model"], self.data["optim"], self.data["loss"]
    
    if self.flag == FLAGS.HEALTH_CODE:
      return self.data

    if self.flag == FLAGS.START_TRAIN:
      return self.data["epochs"], self.data["batch_size"], self.data["data_idxs"], self.data["param"]

    if self.flag == FLAGS.TERMINATE:
      return self.data
    



class Server(object):
  clients = [] # clients must be handled fifo, so use list to contain them
  
  def __init__(self, host, port, max_con):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.bind((host, port))
    self.socket.listen()

  def client_id_to_idx(self, id):
    for idx, client in enumerate(self.clients):
      if client["id"] == id:
        return idx
    return None 

  def __del__(self):
    for client in self.clients:
      client["client"].close()

  def accept(self, id):
    idx = self.client_id_to_idx(id)
    if idx != None: # previous session may still be active, so close the session before starting a new one
      self.clients[idx]["client"].close()

    client, client_addr = self.socket.accept()
    temp = {}
    temp["id"] = id
    temp["client"] = client
    temp["addr"] = client_addr
    self.clients.append(temp)

  def send(self, id, data):
    idx = self.client_id_to_idx(id)
    if idx == None:
      self.accept(id)
      idx = self.client_id_to_idx(id)

    _send(self.clients[idx]['client'], data)
  
  def recv(self, id):
    idx = self.client_id_to_idx(id)
    if idx == None:
      raise Exception('Cannot receive data, no client is connected')
  
    return _recv(self.clients[idx]['client'])

  def send_msg(self, id, flag, data):
    msg = Message(source=-1, flag=flag, data=data)
    self.send(id, msg)

  def close(self, id):
    idx = self.client_id_to_idx(id)
    if idx == None:
      return
    self.clients[idx]['client'].close()
    del self.clients[idx]

    #if self.socket:
    #  self.socket.close()
    #  self.socket = None
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

  def send_msg(self, source, flag, data):
    msg = Message(source=source, flag=flag, data=data)
    self.send(msg)

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


## helper functions ##
def _send(socket, data):
  data = pickle.dumps(data, protocol=3)
  data = pack('>I', len(data)) + data
  socket.sendall(data)

def _recv(socket):
  raw_msglen = recvall(socket, 4)
  if not raw_msglen:
      return None
  msglen = unpack('>I', raw_msglen)[0]
  msg =  recvall(socket, msglen)
  return pickle.loads(msg, encoding="utf-8")
  
def recvall(socket, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = socket.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data