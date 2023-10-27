import torch
from torch import nn
from torch.utils.data import DataLoader

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

from timeit import default_timer as timer
from tqdm.auto import tqdm


def accuracy_fn(y_true, y_pred):
    return (torch.eq(y_true, y_pred).sum().item() / len(y_pred)) * 100


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"

class_names = ["combat", "building", "fire", "rehab", "military"]
BATCH_SIZE = 32
EPOCHS = 100

print(f"Using {device}")
print(f"Class name mapping: {list(enumerate(class_names))}")

# load the image data from "train" folder
train_data = datasets.ImageFolder(
    root="sorted_events",
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize((256, 256)), ToTensor()]
    ),
)
test_data = datasets.ImageFolder(
    root="test",
    transform=torchvision.transforms.Compose(
        [torchvision.transforms.Resize((256, 256)), ToTensor()]
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


class MyCNN(torch.nn.Module):
    def __init__(self, numChannels, classes):
        super(MyCNN, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = torch.nn.Conv2d(
            in_channels=numChannels, out_channels=20, kernel_size=(5, 5)
        )
        self.silu1 = torch.nn.SiLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = torch.nn.Conv2d(
            in_channels=20, out_channels=50, kernel_size=(5, 5)
        )
        self.silu2 = torch.nn.SiLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(in_features=186050, out_features=500)
        self.silu3 = torch.nn.SiLU()
        self.dense1 = torch.nn.Linear(in_features=500, out_features=1024)
        self.dense2 = torch.nn.Linear(in_features=1024, out_features=2048)
        self.dense3 = torch.nn.Linear(in_features=2048, out_features=500)
        self.silu4 = torch.nn.SiLU()
        # initialize our softmax classifier
        self.fc2 = torch.nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.silu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.silu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.silu3(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        x = self.silu4(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output


torch.manual_seed(42)

model = MyCNN(numChannels=3, classes=5).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

train_time_start_on_cpu = timer()

for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n-------")
    epoch_cpu_st = timer()
    ### Training
    train_loss, train_acc = 0, 0
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # model.train()
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss  # accumulatively add up the loss per epoch

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # Print out how many samples have been seen
        # if batch % 160 == 0:
        # print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy
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
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

            # Calculations on test metrics need to happen inside torch.inference_mode()
            # Divide total test loss by length of test dataloader (per batch)
            test_loss /= len(test_dataloader)

            # Divide total accuracy by length of test dataloader (per batch)
            test_acc /= len(test_dataloader)

    ## Print out what's happening
    print(
        f"Train loss: {train_loss:.5f}, Train acc: {train_acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
    )
    epoch_cpu = timer()
    tt = print_train_time(
        start=epoch_cpu_st, end=epoch_cpu, device=str(next(model.parameters()).device)
    )

# Calculate training time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(
    start=train_time_start_on_cpu,
    end=train_time_end_on_cpu,
    device=str(next(model.parameters()).device),
)


torch.manual_seed(42)


def eval_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(
                y_true=y, y_pred=y_pred.argmax(dim=1)
            )  # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_acc": acc,
    }


# Calculate model 0 results on test dataset
model_1_results = eval_model(
    model=model, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn
)
print(model_1_results)

# save model
# torch.save(model.state_dict(), "models/pyTorch-CNN-pjr-2710-033.pt")

model_scripted = torch.jit.script(model)
model_scripted.save("models/torch-CNNv2-2710-01.pt")

# ## LOAD
# # load model
# model = MyCNN(numChannels=3, classes=5).to(device)
# model.load_state_dict(torch.load("models/pyTorch-CNN.pt"))
# model.eval()
