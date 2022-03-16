import argparse
from client import *
import requests
import threading



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="http://127.0.0.1:22222", type=str)
    parser.add_argument("--client", default=5, type=int)
    parser.add_argument("--exp",  default=1, type=int)
    parser.add_argument("--round", default=5, type=int)
    parser.add_argument("--num", default=300, type=int)
    parser.add_argument("--delay", default=5, type=int)
    args = parser.parse_args()

    
    init = requests.get(f"{args.ip}/initialize/{args.client}/{args.exp}/{args.round}")
    print(init)

    clients = []
    for i in range(args.client):
        client = Client(args.ip, args.round, args.delay, args.num, i, args.exp)
        clients.append(client)

    for client in clients:
        thread = threading.Thread(target=client.task)
        thread.start()
        
    print("Federated learning done")