from jetson_fl import FLClient
import threading
import argparse

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--host", default="localhost", type=str)
   parser.add_argument("--port", default=20000, type=int)
   parser.add_argument("--cli", default=5, type=int)
   args = parser.parse_args()
   clients = []
   for i in range(args.cli):
      clients.append(FLClient(i, host=args.host, port=args.port))

   for client in clients:
      thread = threading.Thread(target=client.task)
      thread.start()
