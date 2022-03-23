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

                        
    def start_fed(self, host, port):
        threads = []
        for i, (port, con) in enumerate(zip(self.jetson_ports, self.connections)):
            command = f'docker exec jetson_fl python3 /home/fl_jetson/socket/test_single_client.py --id {i} --host 147.47.200.209 --port 20000'
            print(f'----------------{port}----------------')
            try:
                t=threading.Thread(target=con.run,args=(command,))
                threads.append(t)
                t.start()
                time.sleep(2)
            except:
                print('ERROR')
        
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="147.47.200.209", type=str)
    parser.add_argument("--port", default=20000, type=int)
    parser.add_argument("--min", default=20101, type=int)
    parser.add_argument("--max", default=20136, type=int)

    #args.host = socket.gethostname()
    #print(args.host)
    
    args = parser.parse_args()

    jetson = Jetson(min_port = args.min, max_port=args.max)
    jetson.check() # 통신 전에 무조건 실행되야 함
    
    print("\n")
    print("Stop FL container")
    jetson.send_command("docker stop jetson_fl")
    print("...completed")

    print("\n")
    print("Start FL container")
    jetson.send_command("docker start jetson_fl")
    print("...completed")
    
    #print("\n")
    #print("Pull latest image")
    #jetson.send_command("docker pull crazyboy9103/jetson_fl:latest")
    #print("...completed")
    
    #print("\n")
    #print("Running the container")
    #jetson.send_command("docker run -d -ti --name client --gpus all --network host crazyboy9103/jetson_fl:latest")
    #print("...completed")

    #print("\n")
    #print("Git pull")
    #jetson.send_command("docker exec client cd ambient_fl && git pull")
    #print("...completed")
    
    print("\n")
    print("Starting federated learning")
    jetson.start_fed(host=args.host, port=args.port) 
    
    
    jetson.send_command("docker stop jetson_fl")
    print("Federated learning done")
    