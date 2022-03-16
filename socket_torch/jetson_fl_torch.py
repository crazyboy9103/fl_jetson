from pickle_socket_torch import Client, Message, FLAGS, CLASSIFY_MODELS, DETECT_MODELS, DATASET
import torch, torchvision
import torchvision.models as models
import numpy as np
import logging
import sys
class FLClient:
    def __init__(self, id, host = 'localhost', port = 20000):
        self.id = id
        self.host = host
        self.port = port
        self.logger = self.build_logger("client_log") 
        self.sock_client = Client()
        self.sock_client.connect(id, host, port)
        self.logger.info(f"Connected to {host}:{port} as client {id}")

    def build_logger(self, name):
        logger = logging.getLogger('log_custom')
        logger.setLevel(logging.CRITICAL)

        formatter = logging.Formatter("%(asctime)s;[%(levelname)s];%(message)s",
                              "%Y-%m-%d %H:%M:%S")
        
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(logging.CRITICAL)
        logger.addHandler(streamHandler)

        fileHandler = logging.FileHandler(f'{name}.txt', mode = "a")
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.CRITICAL)
        logger.addHandler(fileHandler)
        
        logger.propagate = False
        return logger
           
    
    def task(self):
        while True:
            msg = self.sock_client.recv()

            if msg.flag == FLAGS.TERMINATE:
                self.repsond_terminate()

            if msg.flag == FLAGS.START_TRAIN:
                self.respond_train(msg)

            if msg.flag == FLAGS.SETUP:
                self.respond_setup(msg) 
        
        
    def repsond_terminate(self):
        self.logger.info(f"connection to {self.host}:{self.port} terminated")
        sys.exit()

    def respond_setup(self, msg):
        self.logger.info(f"client {self.id} setup started")
        print(f"client {self.id} setup started")
        try:
            # 1. gets dataset
            dataset_name, model, optim, loss = msg.get_data()
            self.logger.info(f"client {self.id} config downloaded")
            
            self.trainset, self.testset = self.prepare_dataset(dataset_name)
            self.logger.info(f"client {self.id} dataset prepared")
           
            # 2. builds model from json
#            import threading
#            lock = threading.Lock()
            #model = tf.keras.models.model_from_json(model, custom_objects={"null":None}) 
            self.model = tf.keras.models.model_from_json(model, custom_objects={"null":None})
            self.logger.info(f"client {self.id} model built")

            # 3. compile model 
            self.model.compile(optimizer=optim,loss=loss)
            self.model.train()
            
            # 4. test model
            test_idxs = np.random.choice(len(self.x_train), 100)
            split_x_train, split_y_train = self.x_train[test_idxs], self.y_train[test_idxs]
            self.logger.info(f"client {self.id} test training started")
            print(f"client {self.id} started test training")
#            lock.acquire()
            self.model.fit(split_x_train, split_y_train, epochs=1, batch_size=8, verbose=2)
#            lock.release()
            self.send_msg(flag=FLAGS.SETUP, data=FLAGS.RESULT_OK)
        
        except Exception as e:
            self.logger.critical(f"client {self.id} failed setup")
            self.logger.critical(e)

            self.send_msg(flag=FLAGS.SETUP, data=FLAGS.RESULT_BAD)

    def respond_train(self, msg):
        try:
            print(f"client {self.id} training started")

            self.logger.info(f"client {self.id} training started")
            
            epochs, batch_size, data_idxs, param = msg.get_data()

            self.train_model(data_idxs, param, epochs, batch_size)
            self.send_msg(flag=FLAGS.START_TRAIN, data=updated_param)

            print(f"client {self.id} training completed")

            self.logger.info(f"client {self.id} training completed")

        except Exception as e:
            print(f"client {self.id} training failed")

            self.logger.critical(f"client {self.id} training failed")
            self.logger.critical(e)
            
            self.send_msg(flag=FLAGS.START_TRAIN, data=FLAGS.RESULT_BAD)

    def prepare_dataset(self, name):
        if name == DATASET.MNIST:
            
        if name == DATASET.CIFAR10:
        
        if name == DATASET.CIFAR100:
        
        if name == DATASET.COCO:

        
        return trainset, testset
    

    def build_model(self, model_name):
        if model_name == CLASSIFY_MODELS.MOBILENETV2:
        
        if model_name == CLASSIFY_MODELS.RESNET18:
            
        if model_name == CLASSIFY_MODELS.VGG16:
        
        if model_name == DETECT_MODELS.SSD:
        
        if model_name == DETECT_MODELS.SSDLITE:
            



    def send_msg(self, flag, data=None):
        msg = Message(source=self.id, flag=flag, data=data)
        self.sock_client.send(msg)

    def send_recv_msg(self, flag, data=None):
        msg = Message(source=self.id, flag=flag, data=data)
        self.sock_client.send(msg)
        response = self.sock_client.recv()
        return response

    def train_model(self, data_idxs, global_weight, epochs, batch_size):
        # Train a local model from latest model parameters 
        if global_weight != None:
            global_weight = list(map(lambda weight: np.array(weight), global_weight))
            self.model.set_weights(global_weight)
            
        split_x_train, split_y_train = self.x_train[data_idxs], self.y_train[data_idxs]

        self.model.fit(split_x_train, split_y_train, epochs=epochs, batch_size=batch_size, verbose=0)
