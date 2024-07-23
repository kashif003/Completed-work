from sklearn.model_selection import train_test_split
class GenericSplit:
    """
    A class for splitting data into training, validation, and test sets.
    Attributes:
        data (list or ndarray): The data samples.
        labels (list or ndarray): The labels corresponding to the data samples.
        train_size (int): The proportion of the dataset to include in the training set.
        test_size (int): The proportion of the dataset to include in the test set.
        val_size (int): The proportion of the training set to include in the validation set.
        random_state (int): The seed used by the random number generator.
    """
    def __init__(self, data, labels, train_size, test_size, val_size, random_state=42):
        """
        Initializes the GenericSplit class with the data, labels, and split proportions.

        Args:
            data (list or ndarray): The data samples.
            labels (list or ndarray): The labels corresponding to the data samples.
            train_size (int): The proportion of the dataset to include in the training set.
            test_size (int): The proportion of the dataset to include in the test set.
            val_size (int): The proportion of the training set to include in the validation set.
            random_state (int): The seed used by the random number generator. Defaults to 42.
        """
        self.data = data
        self.labels = labels
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def partitioning(self):
        """
        Splits the data into training, validation, and test sets.

        Returns:
            dict: A dictionary with keys 'train', 'val', and 'test' each containing a list with the corresponding data and labels.
        """
        # First split the data into train+val and test sets
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            self.data, self.labels, test_size=self.test_size, random_state=self.random_state)

        # Then split the train+val set into train and val sets
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_val_data, train_val_labels, test_size=self.val_size, random_state=self.random_state)

        return {'train': [train_data, train_labels], 'val': [val_data, val_labels], 'test': [test_data, test_labels]}

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
class GenericDataset(Dataset):
    """
    A generic dataset class for loading data with PyTorch's DataLoader.

    Attributes:
        data (list or ndarray): The data samples.
        labels (list or ndarray, optional): The labels corresponding to the data samples.
        batch_size (int): The number of samples per batch to load.
        shuffle (bool): Whether to shuffle the data at every epoch.
        drop_last (bool): Whether to drop the last incomplete batch.
        pin_memory (bool): Whether to use pinned (page-locked) memory.
        num_workers (int): How many subprocesses to use for data loading.
        transform (callable, optional): A function/transform to apply to the data.
        data_loader (DataLoader, optional): The DataLoader instance for the dataset.
    """

    def __init__(self, data, labels=None, batch_size=64, shuffle=True, drop_last=True, pin_memory=True, num_workers=2):
        """
        Initializes the GenericDataset.

        Args:
            data (list or ndarray): The data samples.
            labels (list or ndarray, optional): The labels corresponding to the data samples. Defaults to None.
            batch_size (int): The number of samples per batch to load. Defaults to 64.
            shuffle (bool): Whether to shuffle the data at every epoch. Defaults to True.
            drop_last (bool): Whether to drop the last incomplete batch. Defaults to True.
            pin_memory (bool): Whether to use pinned (page-locked) memory. Defaults to True.
            num_workers (int): How many subprocesses to use for data loading. Defaults to 2.
        """
        self.data = data
        self.labels = labels
        self.transform = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.data_loader = None

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the sample (and label if available) at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the sample and its label (or None if labels are not provided).
        """
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        if self.labels is not None:
            label = self.labels[idx]
            return sample, label
        else:
            return sample, None

    def get_data_loader(self, generic_data):
        """
        Creates and returns a DataLoader instance for the dataset.

        Args:
            generic_data (GenericDataset): The dataset instance.

        Returns:
            DataLoader: The DataLoader instance for the dataset.
        """
        self.data_loader = DataLoader(generic_data, batch_size=self.batch_size, shuffle=self.shuffle,
                                      drop_last=self.drop_last, pin_memory=self.pin_memory, num_workers=self.num_workers)
        return self.data_loader

    def set_transform(self, lst_transform):
        """
        Sets the transform to apply to the data.

        Args:
            lst_transform (list): A list of transform functions to apply.
        """
        self.transform = transforms.Compose(lst_transform)


import torch.nn as nn

class GenericLoss(nn.Module):
    """
    A custom loss class that includes L2 regularization.

    Attributes:
        reg_lambda (float): The regularization strength.
        model (nn.Module): The PyTorch model.
        criterion (nn.Module): The primary loss function (CrossEntropyLoss).
    """

    def __init__(self, model, reg_lambda):
        """
        Initializes the GenericLoss class.

        Args:
            model (nn.Module): The PyTorch model.
            reg_lambda (float): The regularization strength.
        """
        super(GenericLoss, self).__init__()
        self.reg_lambda = reg_lambda
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        Computes the loss including L2 regularization.

        Args:
            outputs (torch.Tensor): The model predictions.
            targets (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss = self.criterion(outputs, targets)
        l2_loss = sum(param.pow(2).sum() for param in self.model.parameters())
        loss += self.reg_lambda * l2_loss
        return loss

import os
import torch

class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.

    Attributes:
        filepath (str): Directory where the checkpoint will be saved.
        patience (int): How long to wait after last time validation metric improved.
        minimize (bool): If True, the validation metric is minimized, else it's maximized.
        counter (int): Counts how many epochs have passed since the last improvement.
        best_score (float or None): The best score achieved so far.
        early_stop (bool): Whether to stop the training early.
    """

    def __init__(self, filepath, patience=10, minimize=True):
        """
        Initializes the EarlyStopping object.

        Args:
            filepath (str): Directory where the checkpoint will be saved.
            patience (int): How long to wait after last time validation metric improved. Defaults to 10.
            minimize (bool): If True, the validation metric is minimized, else it's maximized. Defaults to True.
        """
        self.filepath = filepath
        self.patience = patience
        self.minimize = minimize
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if not os.path.isdir(filepath):
            os.makedirs(filepath)
            print(f"Directory not existed! '{filepath}' created.")
        else:
            print(f"Directory existed! '{filepath}'")

    def __call__(self, val_metric, model):
        """
        Checks if the validation metric has improved and updates the early stopping criteria.

        Args:
            val_metric (float): The current validation metric.
            model (nn.Module): The model being trained.
        """
        if self.minimize:
            score = -val_metric
        else:
            score = val_metric
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping")
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, model):
        """
        Saves the model checkpoint.

        Args:
            model (nn.Module): The model to be saved.
        """
        print(f"Saving model with the current best validation metric: {self.best_score}")
        torch.save(model.state_dict(), os.path.join(self.filepath, 'checkpoint.pt'))

class Trainer:

    def __init__(self, model, train_loader, loss_function, optimizer, classes, visualizer):
        self.model = model
        self.train_loader = train_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.classes = classes
        self.visualizer = visualizer


    def train(self, epoch):
        total=0
        true_pred=0
        max_images=6
        running_loss=0.0

        self.model.train()

        device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):

            inputs, targets = inputs.to(device), targets.to(device)

        self.optimizer.zero_grad()

        outputs = self.model(inputs)

        loss = self.loss_function(outputs, targets)

        loss.backward()

        self.optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += targets.size(0)

        true_pred += predicted.eq(targets).sum().item()

        if batch_idx % 128 == 0:
            print(f'Epoch [{epoch+1}], Iteration [{batch_idx+1}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}], Iteration [{batch_idx+1}], Loss: {loss.item():.4f}')
        writer.add_figure('train: PRED vs. GT', self.visualizer.plot_classes_preds(outputs, inputs, targets, self.classes, max_images), global_step=epoch)

        epoch_loss = running_loss / len(self.train_loader)

        epoch_acc = 100.0 * true_pred / total

        writer.add_scalar('training loss', epoch_loss, epoch)
        writer.add_scalar('training accuracy', epoch_acc, epoch)

        return epoch_loss, epoch_acc


class Evaluator:
    def __init__(self, split, model, eval_loader, loss_func, classes, visualizer, ):
        self.split = split
        self.model = model
        self.eval_loader = eval_loader
        self.loss_function = loss_func
        self.classes = classes
        self.visualizer = visualizer

    def evaluate(self, epoch=0):
        total=0
        true_pred=0
        max_images=6
        running_loss=0.0

        self.model.eval()

        device= 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        with torch.no_grad():
            for inputs, targets in self.eval_loader:

                inputs, targets = inputs.to(device), targets.to(device)

                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                eval_loss = loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                true_pred += predicted.eq(targets).sum().item()

                writer.add_figure(self.split+': PRED vs. GT', self.visualizer.plot_classes_preds(outputs, inputs, targets, self.classes, max_images), global_step=epoch)

                epoch_loss = eval_loss / len(self.eval_loader)

                epoch_acc = 100.0 * true_pred / total

                writer.add_scalar(self.split + ' loss', epoch_loss, epoch)
                writer.add_scalar(self.split + ' accuracy', epoch_acc, epoch)

                return epoch_loss, epoch_acc

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
class Visualizer:
    def __init__(self):
        pass

    def matplotlib_imshow(self, img, one_channel=False):

        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5
        img = img.cpu()
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # helper function - taken from https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html, 2.7.24, modified
    def images_to_probs(self, model_outputs):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        _, preds_tensor = torch.max(model_outputs, 1)

        probs = F.softmax(model_outputs, dim=1)

        return preds_tensor, probs

    # helper function - taken from https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html, 2.7.24, modified
    def plot_classes_preds(self, model_outputs, images, labels, classes, max_img):
        '''
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        '''
        preds, probs = self.images_to_probs(model_outputs)

        probs = probs.detach()
        batch_size = preds.shape[0]
        if batch_size >= max_img:
            max_subs = max_img

        fig = plt.figure()
        for idx in np.arange(max_subs):
            ax = fig.add_subplot(1, max_subs, idx + 1, xticks=[], yticks=[])

            self.matplotlib_imshow(torch.clip(images[idx], 0, 1), one_channel=False)

            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format
                (
                classes[preds[idx].item()],
                probs[idx][preds[idx].item()] * 100.0,
                classes[labels[idx].item()]
            ),
                fontsize=8,
                color=("green" if preds[idx] == labels[idx].item() else "red"))
        plt.tight_layout()
        plt.show()
        return fig







