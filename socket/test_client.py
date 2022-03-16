
from pickle_socket import Client, Message
client = Client()
client.connect(id=0, host='127.0.0.1', port=20000) 
#client.send(any python object)
client.send({"test":"test"})
response = client.recv() # blocking
#response도 Message 인스턴스
print("client received", response)

