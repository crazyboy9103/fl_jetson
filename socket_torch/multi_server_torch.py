from pickle_socket_torch import Server, Message, FLAGS, CLASSIFY_MODELS, DETECT_MODELS, DATASET
import numpy as np
import torch, torchvision
import torchvision.models as models
import logging
from datetime import datetime
import threading
import argparse 

class FLServer:
    EXP_UNIFORM = 1
    EXP_RANDOM_SAME_SIZE = 2
    EXP_RANDOM_DIFF_SIZE = 3
    EXP_SKEWED = 4
    
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = Server(host, port, max_con=5)
        self.curr_round = 0
        

    def build_logger(self, name):
        logger = logging.getLogger('log_custom')
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s;[%(levelname)s];%(message)s",
                              "%Y-%m-%d %H:%M:%S")
        
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(logging.INFO)
        logger.addHandler(streamHandler)

        fileHandler = logging.FileHandler(f'log_{name}.txt', mode = "w")
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
        
        logger.propagate = False
        return logger

    def task(self):
        print("Start FedAvg")
        self.logger.info("Start FedAvg")
        # Check round
        if self.curr_round < self.max_round:
            self.curr_round += 1
            self.logger.info(f"Round {self.curr_round}/{self.max_round}")
            print(f"Round {self.curr_round}/{self.max_round}")
        
        else:
            for client in self.server.clients:
                client_id = client["id"]
                self.request_terminate(client_id)
            self.logger.info("Finished FL task")
            print("Finished FL task")
            return 

        # Split dataset 
        self.client_data_idxs = self.split_dataset(self.experiment, self.num_samples)  
        self.logger.info("Started FL task")
        print("Started FL task")
        
        # Gets trained parameters, accuracies
        params, accs = self.train_once(self.epochs, self.batch_size)
        
        # record accs in log
        for id in range(len(accs)):
            self.logger.info(f"client {id} acc {accs[id]}")
            print(f"client {id} acc {accs[id]}")
            
        # FedAvg Algorithm
        N = sum(map(lambda idxs: len(idxs), self.client_data_idxs.values())) 
        for id, idxs in self.client_data_idxs.items():
            temp = {}
            for idx in idxs:
                label = self.y_train[idx]
                if isinstance(label, np.ndarray):
                    label = label[0] 
                if label not in temp:
                    temp[label] = 0
                temp[label] += 1

            temp={k:temp[k] for k in sorted(temp)}
            self.logger.info(f"client {id} data dist {temp}") 

        aggr_layers = {}

        for id, param in params.items():
            n = len(self.client_data_idxs[id])
            self.logger.info(f"client {id}, {n} training data samples")
            print(f"client {id}, {n} training data samples")
            for i, layer in enumerate(param):
                weighted_param = (n / N) * layer
                
                if i not in aggr_layers:
                    aggr_layers[i]  = []
                
                aggr_layers[i].append(weighted_param)
        
        #print(aggr_layers)
        weights = []

        for i, weighted_params in aggr_layers.items():
            block = np.zeros_like(weighted_params[0], dtype=np.float32)
            for param in weighted_params:
                block += param

            weights.append(block)
        
        # swap & evaluate & record server model parameter
        self.model.set_weights(weights)
        acc = self.evaluate_param(id=-1, param=weights, clients_acc_dict={})
        self.logger.info(f"Server acc {acc}")
        print(f"Server acc {acc}")

        return self.task()
        

    def build_model(self, model_type, pretrained=False):
        if model_type == CLASSIFY_MODELS.RESNET18:
            model = models.resnet18(pretrained=pretrained)

        if model_type == CLASSIFY_MODELS.VGG16:
            model = models.vgg16(pretrained=pretrained)

        if model_type == CLASSIFY_MODELS.MOBILENETV2:
            model = models.mobiletnet_v2(pretrained=pretrained)
        if model_type == DETECT_MODELS.SSDLITE:
            return 
        if model_type == DETECT_MODELS.SSD:
            return 
        dataset_type = "mnist" if "mnist" in self.dataset_name else "cifar"
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=self.INPUT_SHAPES[dataset_type]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        return model
    
    def prepare_dataset(self, name):
        # Prepare data in (_, y_train), (x_test, y_test) 
        # y_train on server is used to split dataset according to experiment number
        # x_test, y_test are used to evaluate the aggregated model param
        if name == DATASET.MNIST:
            train_dataset = torchvision.datasets.MNIST(root="./dataset", train=True, download=True)
            test_dataset = torchvision.datasets.MNIST(root="./dataset", test=True, download=True)
        if name == DATASET.CIFAR10:
            train_dataset = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
            test_dataset = torchvision.datasets.CIFAR10(root="./dataset", test=True, download=True)
        if name == DATASET.CIFAR100:
            train_dataset = torchvision.datasets.CIFAR100(root="./dataset", train=True, download=True)
            test_dataset = torchvision.datasets.CIFAR100(root="./dataset", test=True, download=True)
        
        #if name == DATASET.COCO:
        #    train_dataset = torchvision.datasets.MNIST(root="./dataset", train=True, download=True)
        #    test_dataset = torchvision.datasets.MNIST(root="./dataset", test=True, download=True)

        assert train_dataset != None and test_dataset != None
        return train_dataset, test_dataset


    def split_dataset(self, experiment, num_samples):
        min_label = min(self.train_dataset.targets)
        max_label = max(self.train_dataset.targets)
        if isinstance(min_label, torch.Tensor):
            min_label = min_label.item()

        if isinstance(min_label, torch.Tensor):
            min_label = min_label.item()
        
        label_idxs = {label: [] for label in range(min_label, max_label+1)}

        for idx, label in enumerate(self.train_dataset.targets):
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            label_idxs[label].append(idx)
        
       
        all_idxs = [idx for idx in range(len(self.train_dataset.targets))]
        client_data_idxs = {client["id"]: [] for client in self.server.clients}

        num_labels = max_label+1
        if experiment == self.EXP_UNIFORM:
            for label in range(num_labels):
                indices = label_idxs[label]
                for client, data_idxs in client_data_idxs.items():
                    random_idxs = np.random.choice(indices, size=num_samples//num_labels, replace=True).tolist() #bootstrap
                    data_idxs.extend(random_idxs)

        if experiment == self.EXP_RANDOM_SAME_SIZE:
            for client, data_idxs in client_data_idxs.items():
                random_idxs = np.random.choice(all_idxs, size=num_samples, replace=True).tolist()
                data_idxs.extend(random_idxs)

        if experiment == self.EXP_RANDOM_DIFF_SIZE:
            for label in range(num_labels):
                for client, data_idxs in client_data_idxs.items():
                    random_idxs = np.random.choice(all_idxs, size=np.random.randint(1, num_samples//num_labels), replace=True).tolist()
                    data_idxs.extend(random_idxs)

        if experiment == self.EXP_SKEWED:
            all_labels = [label for label in range(num_labels)]
            skewed_labels = np.random.choice(all_labels, np.random.randint(1, num_labels), replace=False)
            non_skewed_labels = set(all_labels)-set(skewed_labels)
            
            for label in skewed_labels:
                for client, data_idxs in client_data_idxs.items():
                    num_data = np.random.randint(int(0.1 * num_samples//num_labels), int(0.3 * num_samples//num_labels))
                    indices = label_idxs[label]
                    random_idxs = np.random.choice(indices, size=num_data, replace=True)
                    data_idxs.extend(random_idxs)
            
            for label in non_skewed_labels:
                for client, data_idxs in client_data_idxs.items():
                    num_data = np.random.randint(int(0.8 * num_samples//num_labels), num_samples//num_labels)
                    indices = label_idxs[label]
                    random_idxs = np.random.choice(indices, size=num_data, replace=True)
                    data_idxs.extend(random_idxs)
           
        return client_data_idxs


    def request_terminate(self, id):
        self.server.send_msg(id=id, flag=FLAGS.TERMINATE, data=None)
        
    def request_train(self, id, epochs, batch_size, clients_param_dict):
        msg = Message(source=-1, flag=FLAGS.START_TRAIN, data={
            "epochs": epochs, 
            "batch_size": batch_size,
            "data_idxs": self.client_data_idxs[id], 
            "param": list(map(lambda layer: layer.tolist(), self.model.get_weights()))
        })
        assert len(msg) != 0, "Message must contain data"
        print("server model parameter size %.2f MB" % (msg.__sizeof__()))
        self.server.send_msg(id=id, flag=FLAGS.START_TRAIN, data={
            "epochs": epochs, 
            "batch_size": batch_size,
            "data_idxs": self.client_data_idxs[id], 
            "param": list(map(lambda layer: layer.tolist(), self.model.get_weights()))
        })
         # uses connection with client and send msg to the client
        recv_msg = self.server.recv(id)
        param = recv_msg.data
        param = list(map(lambda layer: np.array(layer), param))
        clients_param_dict[id] = param
        return param


    def connect(self, id):
        print(f"accepting client {id}")
        self.server.accept(id)
        print(f"client {id} registered")

    def initialize(self, dataset_name, experiment, num_samples, max_clients, max_round, epochs, batch_size):
        # prepare dataset
        self.dataset_name = dataset_name
        print("Prep dataset")
        self.train_dataset, self.test_dataset = self.prepare_dataset(dataset_name)
        
        print("..done")

        # build & compile model 
        print("Build model")
        self.model = self.compile_model(self.build_model())
        print("..done")

        
        # setup logger
        print("Build logger")
        current_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.logger = self.build_logger(current_time)
        print("..done")
        
        
        # setup connections
        print("Build conns")
        #self.client_data_idxs = self.split_dataset(experiment, num_samples) 
        
        threads = []
        for i in range(max_clients):
            thread = threading.Thread(target = self.connect, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print("..done")
        print(len(self.server.clients))

        # variables
        print("Setting up variables")
        self.logger.info(f"max round {max_round}, experiment {experiment}, num samples {num_samples}, epochs {epochs}, batch size {batch_size}")
        self.max_round = max_round
        self.experiment = experiment
        self.num_samples = num_samples
        self.epochs = epochs
        self.batch_size = batch_size
        print("..done")

        print("Health check")
        
        clients_resultcode_dict = {}
        threads = []

        for id in range(max_clients):
            #result_code = self.request_setup(id)
            thread = threading.Thread(target=self.request_setup, args=(id, clients_resultcode_dict,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for id, result_code in clients_resultcode_dict.items():
            healthy = result_code == FLAGS.RESULT_OK
            idx = self.server.client_id_to_idx(id)
            if healthy:
                self.logger.info(f"client {id} address {self.server.clients[idx]['addr']} healthy")
                print(f"client {id} address {self.server.clients[idx]['addr']} healthy")

            else:
                self.logger.info(f"client {id} address {self.server.clients[idx]['addr']} not healthy")
                print(f"client {id} address {self.server.clients[idx]['addr']} not healthy")
                self.request_terminate(id)
                self.server.close(id)
                
        print("..done")
        assert len(self.server.clients) != 0, "no available clients"
        
        print(f"{len(self.server.clients)}/{max_clients} healthy")
        return f"{len(self.server.clients)}/{max_clients} healthy"

    def request_setup(self, id, clients_resultcode_dict):
        msg = Message(source=-1, flag=FLAGS.SETUP, data={
            "dataset_name": self.dataset_name, 
            "model":, 
            "optim":,
            "loss":,
        })
        print("server settings size %.2f MB" % (msg.__sizeof__()))
        assert len(msg) != 0, "Message must contain data"
        self.server.send(id, msg)
        recv_msg = self.server.recv(id)
        result_code = recv_msg.data
        clients_resultcode_dict[id] = result_code
        return result_code
            


    def train_once(self, epochs, batch_size):
        # must run after initialize
        # train and aggregate at once
        clients_param_dict = {}
        clients_acc_dict = {} 
        
        threads = []
        for client in self.server.clients:
            client_id = client["id"]
            thread = threading.Thread(target=self.request_train, args =(client_id, epochs, batch_size, clients_param_dict,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()

        threads = []

        for client in self.server.clients:
            client_id = client["id"]
            param = clients_param_dict[client_id]
            thread = threading.Thread(target=self.evaluate_param, args = (client_id, param, clients_acc_dict, ))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()

        return clients_param_dict, clients_acc_dict

    def evaluate_param(self, id, param, clients_acc_dict):
        temp_param = self.model.get_weights()
        self.model.set_weights(param)

        n = len(self.x_test)
        idxs = np.random.choice(n, n//10, replace=False)
        x_test, y_test = self.x_test[idxs], self.y_test[idxs]

        acc = self.model.evaluate(x_test, y_test)[1]
        clients_acc_dict[id] = acc
        self.model.set_weights(temp_param)
        return acc


    def compile_model(self, model, optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="experiment 1, 2, 3, 4", type=int, default=3)
    parser.add_argument("--num", help="num_samples", type=int, default=200)
    parser.add_argument("--cli", help="max_clients", type=int, default=2)
    parser.add_argument("--round", help="max_round", type=int, default=5)
    parser.add_argument("--data", help="dataset_name", type=str, default="mnist")
    parser.add_argument("--host", help="host", type=str, default="127.0.0.1")
    parser.add_argument("--port", help="port", type=int, default=20000)
    parser.add_argument("--epochs", help="epochs", type=int, default=10)
    parser.add_argument("--batch", help="batch_size", type=int, default=8)

    args = parser.parse_args()

    
    #args.host = socket.gethostname()
    #print(args.host)
    FL_Server = FLServer(host=args.host, port=args.port) 
    print(FL_Server.initialize(args.data, args.exp, args.num, args.cli, args.round, args.epochs, args.batch))
    FL_Server.task()