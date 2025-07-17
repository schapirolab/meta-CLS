import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataloaders.idataset import DummyArrayDataset
import os


class IncrementalLoader:

    def __init__(
        self,
        opt,
        shuffle=True,
        seed=1,
    ):
        # opt is equivalent to args: can do opt.add_item_labels
        self._opt = opt
        validation_split=opt.validation
        increment=opt.increment

        if self._opt.num_of_item_labels != 0:
            self._item_labels = []
            self._alt_test_dataset = []

        self._setup_data(
            class_order_type=opt.class_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split
        )

        self._current_task = 0

        self._batch_size = opt.batch_size
        self._test_batch_size = opt.test_batch_size        
        self._workers = opt.workers
        self._shuffle = shuffle
        
        self._setup_test_tasks()

    @property
    def n_tasks(self):
        if self._opt.num_of_item_labels != 0:
            return len(self._alt_test_dataset)
        else:
            return len(self.test_dataset)
    
    def new_task(self):
        if self._opt.num_of_item_labels != 0:
            if self._current_task >= len(self._alt_test_dataset):
                raise Exception("No more tasks.")
        else:
            if self._current_task >= len(self.test_dataset):
                raise Exception("No more tasks.")

        p = self.sample_permutations[self._current_task]
        x_train, y_train = self.train_dataset[self._current_task][1][p], self.train_dataset[self._current_task][2][p]
        x_test, y_test = self.test_dataset[self._current_task][1], self.test_dataset[self._current_task][2]

        if self._opt.num_of_item_labels != 0:

            if self._opt.item_option == "per_item":
                z_train = self._item_labels[self._current_task]
            else:
                z_train = self._item_labels[self._current_task][p]
            x_test, y_test, z_test = x_train, y_train, z_train

            train_loader = self._get_loader(x_train, y_train, z=z_train, mode="train")
            test_loader = self._get_loader(x_test, y_test, z=z_test, mode="test")

        else:

            train_loader = self._get_loader(x_train, y_train, mode="train")
            test_loader = self._get_loader(x_test, y_test, mode="test")

        task_info = {
            "min_class": 0,
            "max_class": self.n_outputs,
            "increment": -1,
            "task": self._current_task,
            "max_task": len(self.test_dataset),
            "n_train_data": len(x_train),
            "n_test_data": len(x_test)
        }

        self._current_task += 1

        return task_info, train_loader, None, test_loader

    def _setup_test_tasks(self):
        self.test_tasks = []
        for i in range(len(self.test_dataset)):
            self.test_tasks.append(self._get_loader(self.test_dataset[i][1], self.test_dataset[i][2], mode="test"))

    def get_tasks(self, dataset_type='test'):
        if self._opt.num_of_item_labels != 0:
            if dataset_type == 'test':
                return self._alt_test_dataset
            elif dataset_type == 'val':
                return self._alt_test_dataset
            else:
                raise NotImplementedError("Unknown mode {}.".format(dataset_type))
        else:
            if dataset_type == 'test':
                return self.test_dataset
            elif dataset_type == 'val':
                return self.test_dataset
            else:
                raise NotImplementedError("Unknown mode {}.".format(dataset_type))

    def get_dataset_info(self):
        n_inputs = self.train_dataset[0][1].size(1)
        n_outputs = 0
        for i in range(len(self.train_dataset)):
            n_outputs = max(n_outputs, self.train_dataset[i][2].max())
            if self._opt.num_of_item_labels == 0:
                n_outputs = max(n_outputs, self.test_dataset[i][2].max())
        self.n_outputs = n_outputs 

        return n_inputs, n_outputs.item()+1, self.n_tasks


    def _get_loader(self, x, y, z=None, shuffle=True, mode="train"):
        

        if mode == "train":
            batch_size = self._batch_size
        elif mode == "test":
            batch_size = self._test_batch_size
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))


        if self._opt.num_of_item_labels != 0 :
            return DataLoader(
                DummyArrayDataset(x, y, z=z, n_item_labels=self._opt.num_of_item_labels),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self._workers
            )
        else:
            return DataLoader(
                DummyArrayDataset(x, y),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self._workers
            )



    def _setup_data(self, class_order_type=False, seed=1, increment=10, validation_split=0.):
        # FIXME: handles online loading of images
        torch.manual_seed(seed)


        if self._opt.dataset == "fashion_mnist":

            transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
            # self.fashion_train_dataset = datasets.FashionMNIST('F_MNIST_data', download=True, train=True, transform=transform)
            # self.fashion_test_dataset = datasets.FashionMNIST('F_MNIST_data', download=True, train=False, transform=transform)

            self.fashion_train_dataset = torch.load(os.path.join(self._opt.data_path, "training.pt"))
            self.fashion_test_dataset = torch.load(os.path.join(self._opt.data_path, "test.pt"))

            self.train_dataset = []
            self.test_dataset = []

            if self._opt.small_test:
                self.raw_test_dataset = []

            n_of_tasks = 5
            for i in range(n_of_tasks):
                train_indices = torch.where((self.fashion_train_dataset[:][1] == i*2) | (self.fashion_train_dataset[:][1] == i*2+1))[0]
                
                train_x_temp = torch.index_select(self.fashion_train_dataset[0], 0, torch.LongTensor(train_indices)).float()
                train_x_temp = torch.reshape(train_x_temp, (train_x_temp.size(0), train_x_temp.size(1)*train_x_temp.size(2)))

                train_y_temp = torch.index_select(self.fashion_train_dataset[1], 0, torch.LongTensor(train_indices))

                self.train_dataset.append([0.0, train_x_temp, train_y_temp])

                test_indices = torch.where((self.fashion_test_dataset[1] == i*2) | (self.fashion_test_dataset[1] == i*2+1))[0]
                
                test_x_temp = torch.index_select(self.fashion_test_dataset[0], 0, torch.LongTensor(test_indices)).float()
                test_x_temp = torch.reshape(test_x_temp, (test_x_temp.size(0), test_x_temp.size(1)*test_x_temp.size(2)))

                test_y_temp = torch.index_select(self.fashion_test_dataset[1], 0, torch.LongTensor(test_indices))

                if self._opt.small_test:
                    self.raw_test_dataset.append([0.0, test_x_temp, test_y_temp])
                else:
                    self.test_dataset.append([0.0, test_x_temp, test_y_temp])

        else:
            if self._opt.small_test:
                self.train_dataset, self.raw_test_dataset = torch.load(os.path.join(self._opt.data_path, self._opt.dataset + ".pt"))
                self.test_dataset = []
            else:
                self.train_dataset, self.test_dataset = torch.load(os.path.join(self._opt.data_path, self._opt.dataset + ".pt"))

        self.sample_permutations = []
        self.test_permutations = []

        for t in range(len(self.train_dataset)):
            N = self.train_dataset[t][1].size(0)
            if self._opt.samples_per_task <= 0:
                n = N
            else:
                n = min(self._opt.samples_per_task, N)


            p = torch.randperm(N)[0:n]
            self.sample_permutations.append(p)

            if self._opt.small_test:
                N_test = self.raw_test_dataset[t][1].size(0)
            else:
                N_test = self.test_dataset[t][1].size(0)

            if self._opt.samples_per_task <= 0:
                n_test = N_test
            else:
                n_test = min(500, N_test)
            p_test = torch.randperm(N_test)[0:n_test]
            self.test_permutations.append(p_test)

            task_index = t
            if self._opt.num_of_item_labels != 0:
                if self._opt.item_option == "per_item":
                    item_labels = np.arange(0+self._opt.samples_per_task*task_index, self._opt.samples_per_task*(task_index+1))
                    np.random.shuffle(item_labels)
                    self._item_labels.append(torch.from_numpy(
                        item_labels
                    ))
                elif self._opt.item_option == "set_label":
                    item_labels = np.arange(0, self._opt.samples_per_task)
                    np.random.shuffle(item_labels)
                    self._item_labels.append(torch.from_numpy(
                        item_labels
                    ))
                else:
                    self._item_labels.append(torch.from_numpy(
                        np.random.randint(0+self._opt.num_of_item_labels*task_index, self._opt.num_of_item_labels*(task_index+1), self.train_dataset[task_index][2].size(0))
                    ))


        if self._opt.num_of_item_labels != 0:
            for task_index in range(len(self.sample_permutations)):
                p = self.sample_permutations[task_index]
                x_train, y_train = self.train_dataset[task_index][1][p], self.train_dataset[task_index][2][p]
                if self._opt.item_option == "per_item" or self._opt.item_option == "set_label":
                    z_train = self._item_labels[task_index]
                else:
                    z_train = self._item_labels[task_index][p]

                self._alt_test_dataset.append([self.train_dataset[task_index][0], x_train, y_train, z_train])
            self.test_dataset = self._alt_test_dataset
        else:
            if self._opt.small_test:
                for task_index in range(len(self.test_permutations)):
                    p = self.test_permutations[task_index]
                    x_test, y_test = self.raw_test_dataset[task_index][1][p], self.raw_test_dataset[task_index][2][p]

                    self.test_dataset.append([self.raw_test_dataset[task_index][0], x_test, y_test])

