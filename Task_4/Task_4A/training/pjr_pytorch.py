from datetime import datetime
import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

from timeit import default_timer as timer
from tqdm.auto import tqdm


def accuracy_fn(y_true, y_pred):
    return (torch.eq(y_true, y_pred).sum().item() / len(y_pred)) * 100


def print_train_time(start: float, end: float, device: torch.device = None):  # type: ignore
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")


def load_data():
    # load the image data from "train" folder
    train_data = datasets.ImageFolder(
        root=TRAIN_FOLDER,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize((INPUT_SIZE, INPUT_SIZE)), ToTensor()]
        ),
    )
    test_data = datasets.ImageFolder(
        root=TEST_FOLDER,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize((INPUT_SIZE, INPUT_SIZE)), ToTensor()]
        ),
    )

    train_dataloader = DataLoader(
        train_data,  # dataset to turn into iterable
        batch_size=BATCH_SIZE,  # how many samples per batch?
        shuffle=True,  # shuffle data every epoch?
    )

    test_dataloader = DataLoader(
        test_data,  # dataset to turn into iterable
        batch_size=BATCH_SIZE,  # how many samples per batch?
        shuffle=True,  # shuffle data every epoch?
    )

    return train_dataloader, test_dataloader


def create_model(device):
    class MyCNN(torch.nn.Module):
        def __init__(self, numChannels, classes):
            super(MyCNN, self).__init__()

            self.cnn_layers = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=numChannels, out_channels=12, kernel_size=(5, 5)
                ),
                torch.nn.BatchNorm2d(num_features=12),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3)),
                torch.nn.BatchNorm2d(num_features=12),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            )

            # initialize first (and only) set of FC => RELU layers

            self.linear_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=(12 * 16 * 16), out_features=classes),
            )

        def forward(self, x):
            # pass the input through our first set of CONV => RELU =>
            # POOL layers
            x = self.cnn_layers(x)
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)

            return x

    model = MyCNN(numChannels=3, classes=5).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

    print(model)

    return model, loss_fn, optimizer


def train(train_dataloader, device, model, loss_fn, optimizer, test_dataloader):
    torch.manual_seed(42)
    full_train_start = timer()
    for epoch in tqdm(range(EPOCHS)):
        print(f"Epoch: {epoch}\n-------")
        epoch_st = timer()
        ### Training
        train_loss, train_acc = 0, 0
        # Add a loop to loop through training batches
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)
            # 2. Calculate loss (per batch)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            # 3. Optimizer zero grad
            optimizer.zero_grad()
            # 4. Loss backward
            loss.backward()
            # 5. Optimizer step
            optimizer.step()
            # print(y_pred.argmax(dim=1))
            train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        ### Testing
        test_loss, test_acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                test_pred = model(X)
                # 2. Calculate loss (accumatively)
                test_loss += loss_fn(
                    test_pred, y
                )  # accumulatively add up the loss per epoch
                # 3. Calculate accuracy (preds need to be same as y_true)
                # print("test", test_pred.argmax(dim=1))
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

        print(
            f"Train loss: {train_loss:.5f}, Train acc: {train_acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
        )

        epoch_cpu = timer()
        print_train_time(start=epoch_st, end=epoch_cpu, device=device)

    # Calculate training time
    full_train_end = timer()
    print_train_time(start=full_train_start, end=full_train_end, device=device)


def eval_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,  # type: ignore
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device,
):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),  # type: ignore
        "model_acc": acc,
    }


def save_model(model: torch.nn.Module):
    # save model
    model_scripted = torch.jit.script(model)  # type: ignore
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Saving to {MODEL_NAME}{INPUT_SIZE}_{current_timestamp}.pt")
    model_scripted.save(f"{MODEL_NAME}{INPUT_SIZE}_{current_timestamp}.pt")  # type: ignore


CLASS_NAMES = [
    "combat",
    "destroyed_buildings",
    "fire",
    "human_aid_rehabilitation",
    "military_vehicles",
]
BATCH_SIZE = 32
EPOCHS = 100
TRAIN_FOLDER = "pjrtest"
TEST_FOLDER = "pjrtest"
MODEL_NAME = "pjr_model_overfit_"
INPUT_SIZE = 75


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # type: ignore # MacOS
    # device = "cuda" if torch.cuda.is_available() else "cpu" # Windows
    # device = "cpu" # other systems 

    print(f"Using {device}")
    print(f"Class name mapping: {list(enumerate(CLASS_NAMES))}\n\n")

    trainloader, testloader = load_data()
    model, loss_fn, optimizer = create_model(device=device)
    train(trainloader, device, model, loss_fn, optimizer, testloader)
    print(eval_model(model, testloader, loss_fn, accuracy_fn, device))
    save_model(model)


if __name__ == "__main__":
    main()
