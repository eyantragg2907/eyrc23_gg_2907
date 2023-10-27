import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision

BATCH_SIZE = 32

torch.manual_seed(42)

def accuracy_fn(y_true, y_pred):
    return (torch.eq(y_true, y_pred).sum().item() / len(y_pred)) * 100


class MyCNN(torch.nn.Module):
	def __init__(self, numChannels, classes):
		super(MyCNN, self).__init__()
        
		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = torch.nn.Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1 = torch.nn.ReLU()
		self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = torch.nn.ReLU()
		self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
		self.flatten = torch.nn.Flatten()
		self.fc1 = torch.nn.Linear(in_features=186050, out_features=500)
		self.relu3 = torch.nn.ReLU()
		# initialize our softmax classifier
		self.fc2 = torch.nn.Linear(in_features=500, out_features=classes)
		self.logSoftmax = torch.nn.LogSoftmax(dim=1)

	def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = self.flatten(x)
		x = self.fc1(x)
		x = self.relu3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)
		# return the output predictions
		return output


model = MyCNN(numChannels=3, classes=5)
model.load_state_dict(torch.load("models/pyTorch-CNN-2710-01.pt"))
loss_fn = torch.nn.CrossEntropyLoss()

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            print(y)
            # break
            y_pred = model(X)
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}

train_data = datasets.ImageFolder(root="train", transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)), torchvision.transforms.ToTensor()]))
test_data = datasets.ImageFolder(root="test", transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)), torchvision.transforms.ToTensor()]))

train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

# Calculate model 0 results on train dataset
# model_1_results = eval_model(model=model, data_loader=train_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
# torch.Size([32, 3, 256, 256]) torch.Size([32])
print(model)
# print(model_1_results)

# EVALUATE ONLY ON ONE IMAGE (FOR TESTING)
img = torchvision.io.read_image("test/1/building1.jpeg", torchvision.io.image.ImageReadMode.RGB).float()
# apply transforms
img = torchvision.transforms.Compose(
      [torchvision.transforms.Resize((256,256), antialias=True)]
)(img)

x = torch.unsqueeze(img, 0)
y = torch.Tensor([0])

model.eval()
with torch.inference_mode():
    y_pred = model(x)

print(int(y_pred.argmax(dim=1)[0]))