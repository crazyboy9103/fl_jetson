from pickle_socket import Server, Message
server = Server(host='192.168.1.157', port=20000, max_con = 5)
server.accept(id = 0)
response = server.recv(id = 0) # Message 객체
server.send(id = 0, data="received") # data를 pickle로 serialize시킨후 전송
print("server received", response)
