import argparse
import threading
import fabric
from fabric import Connection
from random import random
import paramiko
import time


class Jetson:
    def __init__(self, min_port, max_port):
        self.address = "147.47.200.209"
        self.username, self.password = "jetson", "jetson"
        self.ports = [i for i in range(int(min_port), int(max_port)+1) if 1<=i%10<=6]
        self.jetson_ports = []
        self.connections = []
        
    def check(self):
        for port in self.ports:
            con = Connection(f'{self.username}@{self.address}:{port}', connect_kwargs ={"password":self.password})
            command = 'ls'
            print(f'----------------{port}----------------')
            try:
                con.run(command)
                self.jetson_ports.append(port)
                self.connections.append(con)
            except:
                print('ERROR')

        print("Available ports", self.jetson_ports)
            
    
    
    def send_command(self, command):
        for port, con in zip(self.jetson_ports, self.connections): 
            print(f'----------------{port}----------------')
            try:
                con.run(command)

            except:
                print('ERROR')

                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="147.47.200.209", type=str)
    parser.add_argument("--port", default=20000, type=int)
    parser.add_argument("--min", default=20101, type=int)
    parser.add_argument("--max", default=20136, type=int)
  
    args = parser.parse_args()

    jetson = Jetson(min_port = args.min, max_port=args.max)
    jetson.check() # 통신 전에 무조건 실행되야 함
    
    #print("\n")
    #print("Kill all containers")
    #jetson.send_command("docker stop $(docker ps -q)")
    #print("...completed")
    while True:
        code_to_exec = input("명령어 입력 : ")
        if not code_to_exec:
            break
        
        jetson.send_command(code_to_exec)
        
