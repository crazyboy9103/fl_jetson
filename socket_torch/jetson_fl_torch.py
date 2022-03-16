from pickle_socket_torch import Client, Message, FLAGS, CLASSIFY_MODELS, DETECT_MODELS
import torch, torchvision
import torchvision.models as models
import numpy as np


class FLClient:
    def __init__(self, id, host = 'localhost', port = 20000):
        self.id = id
        self.host = host
        self.port = port
        self.sock_client = Client()
        self.sock_client.connect(id, host, port)
       
     
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
        return 

    def respond_setup(self, msg):
        print(f"client {self.id} setup started")
        try:
            # 1. gets dataset
            data = msg.data
            dataset_name = data['dataset_name']
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.prepare_dataset(dataset_name)
           
           
            # 2. builds model from json

            model_arch = data['arch']
            import threading
            lock = threading.Lock()
            #model = tf.keras.models.model_from_json(model, custom_objects={"null":None}) 
            self.model = tf.keras.models.model_from_json(model_arch, custom_objects={"null":None})

            print("model arch", model_arch)

            # 3. compile model 
            optimizer, loss, metrics = tf.keras.optimizers.deserialize(data['optim']), tf.keras.losses.deserialize(data['loss']), data['metrics']
            print("optim, loss, metrics", data['optim'], data["loss"], data["metrics"])
            self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
            
            # 4. test model
            test_idxs = np.random.choice(len(self.x_train), 100)
            split_x_train, split_y_train = self.x_train[test_idxs], self.y_train[test_idxs]

            print(f"client {self.id} started test training")
            print(len(split_x_train))
            lock.acquire()
            self.model.fit(split_x_train, split_y_train, epochs=1, batch_size=8, verbose=2)
            lock.release()
            self.sock_client.send_msg(source=self.id, flag=FLAGS.HEALTH_CODE, data=FLAGS.RESULT_OK)
        
        except Exception as e:
            print(e)
            self.sock_client.send_msg(source=self.id, flag=FLAGS.HEALTH_CODE, data=FLAGS.RESULT_BAD)

    def respond_train(self, msg):
        print(f"client {self.id} training started")
        data = msg.data
        data_idxs = data['data_idxs']
        param = data['param']
        epochs = data['epochs']
        batch_size = data['batch_size']

        self.train_model(data_idxs, param, epochs, batch_size)
        
        self.sock_client.send_msg(source=self.id, flag=FLAGS.START_TRAIN, data=)
        print(f"client {self.id} training completed")


    def prepare_dataset(self, name):
        if name == DATASET.MNIST:
            
        if name == DATASET.CIFAR10:
        
        if name == DATASET.CIFAR100:
        
        if name == DATASET.COCO:


    




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
