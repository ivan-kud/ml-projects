import time

from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, SequentialSampler


def get_mean_and_std(dataset, img_height, img_width, channel_num, batch_size):
    # Initialize variables
    pixels = img_height * img_width
    channels_sum = torch.zeros(channel_num, dtype=torch.double)
    channels_squared_sum = torch.zeros(channel_num, dtype=torch.double)

    # Cycle over batches
    for i, (batch, _) in enumerate(
            tqdm(DataLoader(dataset, batch_size=batch_size))):
        # Sum over batch, height and width, but not over the channels
        channels_sum += torch.sum(batch, dim=[0, 2, 3])
        channels_squared_sum += torch.sum(batch ** 2, dim=[0, 2, 3])

    # mean = E[X]
    mean = channels_sum / (len(dataset) * pixels)

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / (len(dataset) * pixels) - mean ** 2) ** 0.5

    return mean.item(), std.item()


def split_xy_dataset(dataset):
    x = torch.empty(*((len(dataset),) + dataset[0][0].shape),
                    dtype=dataset[0][0].dtype)
    y = torch.empty(len(dataset))
    for i, (x_, y_) in enumerate(dataset):
        x[i] = x_
        y[i] = y_
    return x, y


def merge_xy_dataset(x_dataset, y_dataset):
    dataset = []
    for xy in zip(x_dataset, y_dataset):
        dataset.append(xy)
    return dataset


def _timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()

        # Print elapsed time
        time.sleep(0.1)  # for clear printing
        elapsed_time = end_time - start_time
        print('Training complete in',
              f'{elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s',
              sep=' ')
    return wrapper


class WrappedNN:
    def __init__(self, model, loss_fn, optimizer, scheduler=None, writer=None,
                 device=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self._clear_history()
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Move model parameters and buffers to device
        self.model.to(self.device)
        print(f'Using {self.device} device')

    def _clear_history(self):
        self.train_history = {
            'train average loss': [],
            'train average accuracy': [],
            'valid loss': [],
            'valid accuracy': [],
        }
        self.lr_history = []

    def _train_epoch(self, dataloader, epoch, epochs):
        # Enter training mode
        self.model.train()

        # Initialize variables
        num_samples = len(dataloader.dataset)
        num_batches = len(dataloader)
        acc_loss, acc_accuracy = 0., 0
        current_sample = 0

        # Cycle over batches
        loop = tqdm(enumerate(dataloader), f'Epoch [{epoch + 1}/{epochs}]',
                    num_batches)
        for batch, (x, y) in loop:
            x, y = x.to(self.device), y.to(self.device)
            batch += 1

            # Compute prediction error
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate loss and accuracy
            acc_loss += loss.item()
            acc_accuracy += (pred.argmax(1) == y).sum().item()
            current_sample += len(x)

            # Print current results
            if batch % 100 == 0 or batch == num_batches:
                loop.set_postfix(
                    avg_loss=f'{(acc_loss / batch):>7f}',
                    avg_accuracy=f'{(acc_accuracy / current_sample):>7f}'
                )

            # Step LR scheduler
            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.lr_history.append(self.scheduler.get_last_lr()[0])
                self.scheduler.step(epoch + batch/num_batches)

        # Store average loss and accuracy
        avg_loss = acc_loss / num_batches
        avg_accuracy = acc_accuracy / num_samples
        self.train_history['train average loss'].append(avg_loss)
        self.train_history['train average accuracy'].append(avg_accuracy)
        if self.writer is not None:
            self.writer.add_scalars('loss', {'train': avg_loss},
                                    epoch + 1)
            self.writer.add_scalars('accuracy', {'train': avg_accuracy},
                                    epoch + 1)

    def _eval(self, dataloader, epoch):
        # Enter evaluation mode
        self.model.eval()

        # Initialize variables
        num_samples = len(dataloader.dataset)
        num_batches = len(dataloader)
        acc_loss, acc_accuracy = 0., 0

        with torch.no_grad():
            # Cycle over batches
            loop = tqdm(enumerate(dataloader), 'Validation', num_batches)
            for batch, (x, y) in loop:
                x, y = x.to(self.device), y.to(self.device)
                batch += 1

                # Compute prediction and accumulate loss
                pred = self.model(x)
                acc_loss += self.loss_fn(pred, y).item()

                # Accumulate accuracy
                acc_accuracy += (pred.argmax(1) == y).sum().item()

                # Print results
                if batch == num_batches:
                    loop.set_postfix(
                        loss=f'{(acc_loss / num_batches):>7f}',
                        accuracy=f'{(acc_accuracy / num_samples):>7f}'
                    )

        # Store loss and accuracy
        loss = acc_loss / num_batches
        accuracy = acc_accuracy / num_samples
        self.train_history['valid loss'].append(loss)
        self.train_history['valid accuracy'].append(accuracy)
        if self.writer is not None:
            self.writer.add_scalars('loss', {'valid': loss}, epoch + 1)
            self.writer.add_scalars('accuracy', {'valid': accuracy}, epoch + 1)

    def _step_scheduler(self, epoch):
        if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
            if (hasattr(self.scheduler, 'gamma')
                    and self.scheduler.T_cur == 0.0):
                self.scheduler.base_lrs = [
                    x * self.scheduler.gamma
                    for x in self.scheduler.base_lrs
                ]
                self.scheduler.step(epoch + 1)
        elif self.scheduler is not None:
            self.lr_history.append(self.scheduler.get_last_lr()[0])
            self.scheduler.step()

    @_timer
    def train(self, train_dataloader, valid_dataloader, epochs):
        # Cycle over epochs
        for epoch in range(epochs):
            # Train and evaluate the model
            self._train_epoch(train_dataloader, epoch, epochs)
            self._eval(valid_dataloader, epoch)

            # Step LR scheduler
            self._step_scheduler(epoch)

        # Close tensorboard writer
        if self.writer is not None:
            self.writer.close()

    def predict_proba(self, x_dataset, batch_size=64):
        # Define dataloader
        sampler = SequentialSampler(x_dataset)
        dataloader = DataLoader(x_dataset, batch_size, sampler=sampler)

        # Form empty output tensor
        num_classes = self.model(torch.unsqueeze(x_dataset[0], dim=0)).shape[1]
        pred = torch.empty(len(x_dataset), num_classes)

        # Enter inference mode and cycle over batches
        with torch.inference_mode():
            for i, x in enumerate(tqdm(dataloader)):
                x = x.to(self.device)

                # Compute prediction
                pred_batch = torch.nn.Softmax(dim=1)(self.model(x))
                pred[i * batch_size:(i+1) * batch_size] = pred_batch
        return pred

    def predict(self, x_dataset, batch_size=64):
        return self.predict_proba(x_dataset, batch_size).argmax(1)

    def score(self, x_dataset, y_dataset, batch_size=64):
        y_dataset = torch.Tensor(y_dataset)
        pred = self.predict(x_dataset, batch_size)
        return (pred == y_dataset).sum().item() / len(y_dataset)


if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import train_test_split
    from torch.optim import SGD
    from torch.utils.data import Subset
    import torchvision
    from torchvision.transforms import Compose, ToTensor, Normalize

    VALID_SIZE = 0.2
    BATCH_SIZE = 64
    RANDOM_STATE = 2147483647
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    DATA_PATH = 'data/SVHN'
    data_mean, data_std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]

    # Training dataset
    training_dataset = torchvision.datasets.SVHN(
        root=DATA_PATH,
        split='train',
        download=True,
        transform=Compose([
            ToTensor(),
            Normalize(data_mean, data_std),
        ]))

    # Test dataset
    test_dataset = torchvision.datasets.SVHN(
        root=DATA_PATH,
        split='test',
        download=True,
        transform=Compose([
            ToTensor(),
            Normalize(data_mean, data_std),
        ]))

    # Split indices on train and validation parts
    train_idx, valid_idx = train_test_split(
        np.arange(len(training_dataset)),
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=training_dataset.labels,
    )

    # Define train and validation datasets
    train_dataset = Subset(training_dataset, train_idx)
    valid_dataset = Subset(training_dataset, valid_idx)
    del training_dataset

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE)
    valid_dataloader = DataLoader(valid_dataset, BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE)


    class DenseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            self.linear_relu_stack = torch.nn.Sequential(
                torch.nn.Linear(3 * IMG_HEIGHT * IMG_WIDTH, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


    epochs = 3
    lr = 1e-1

    model = DenseModel()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
    scheduler.gamma = 0.5

    model = WrappedNN(model, loss_fn, optimizer, scheduler)
    print(model.model)

    model.train(train_dataloader, valid_dataloader, epochs)
