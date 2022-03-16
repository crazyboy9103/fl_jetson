from jetson_fl import FLClient
import threading
import argparse

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--host", default="localhost", type=str)
   parser.add_argument("--port", default=20000, type=int)
   parser.add_argument("--id", default=0, type=int)
   args = parser.parse_args()
   client = FLClient(args.id, host=args.host, port=args.port)
   client.task()

