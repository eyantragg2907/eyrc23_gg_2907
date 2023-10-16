'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			GG_2907
# Author List:		Arnav Rustagi (@thearnavrustagi), Pranjal Rastogi (@PjrCodes)
# Filename:			task_1a.py
# Functions:	    [`identify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas
import torch
import numpy
###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################
class Data(torch.utils.data.Dataset):
	'''
	Utility class subclassing torch.Dataset for our use case
	'''
	def __init__(self, X_train, y_train):
		self.X = X_train
		self.y = y_train
		self.len = self.X.shape[0]

	def __getitem__(self, index):
		return self.X[index], self.y[index]

	def __len__(self):
		return self.len


##############################################################


def data_preprocessing(task_1a_dataframe):

	''' 
	Purpose:
	---
	This function will be used to load your csv dataset and preprocess it.
	Preprocessing involves cleaning the dataset by removing unwanted features,
	decision about what needs to be done with missing values etc. Note that 
	there are features in the csv file whose values are textual (eg: Industry, 
	Education Level etc)These features might be required for training the model
	but can not be given directly as strings for training. Hence this function 
	should return encoded dataframe in which all the textual features are 
	numerically labeled.
	
	Input Arguments:
	---
	`task_1a_dataframe`: [Dataframe]
						  Pandas dataframe read from the provided dataset 	
	
	Returns:
	---
	`encoded_dataframe` : [ Dataframe ]
						  Pandas dataframe that has all the features mapped to 
						  numbers starting from zero

	Example call:
	---
	encoded_dataframe = data_preprocessing(task_1a_dataframe)
	'''

	#################	ADD YOUR CODE HERE	##################
	labels = task_1a_dataframe.copy()
	# Simple label encoding using sklearn
	labels["EverBenched"] = LabelEncoder().fit_transform(labels["EverBenched"])
	labels["Gender"] = LabelEncoder().fit_transform(labels["Gender"])
	labels["Education"] = LabelEncoder().fit_transform(labels["Education"])
	labels["City"] = LabelEncoder().fit_transform(labels["City"])
	encoded_dataframe = labels
	##########################################################

	return encoded_dataframe


def identify_features_and_targets(encoded_dataframe):
	'''
	Purpose:
	---
	The purpose of this function is to define the features and
	the required target labels. The function returns a python list
	in which the first item is the selected features and second 
	item is the target label

	Input Arguments:
	---
	`encoded_dataframe` : [ Dataframe ]
						Pandas dataframe that has all the features mapped to 
						numbers starting from zero
	
	Returns:
	---
	`features_and_targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label

	Example call:
	---
	features_and_targets = identify_features_and_targets(encoded_dataframe)
	'''
	#################	ADD YOUR CODE HERE	##################
	dfcopy = encoded_dataframe.copy()
	labels = dfcopy.pop("LeaveOrNot")
	# scaling the input helps the model accuracy because it's binary classification
	scaler = StandardScaler() 
	dfcopy = scaler.fit_transform(dfcopy)  # scaling only the features
	dfcopy = pandas.DataFrame(dfcopy, columns=encoded_dataframe.columns[:-1])
	features_and_targets = [dfcopy, labels]
	##########################################################

	return features_and_targets


def load_as_tensors(features_and_targets):
	
	''' 
	Purpose:
	---
	This function aims at loading your data (both training and validation)
	as PyTorch tensors. Here you will have to split the dataset for training 
	and validation, and then load them as as tensors. 
	Training of the model requires iterating over the training tensors. 
	Hence the training sensors need to be converted to iterable dataset
	object.
	
	Input Arguments:
	---
	`features_and targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label
	
	Returns:
	---
	`tensors_and_iterable_training_data` : [ list ]
											Items:
											[0]: X_train_tensor: Training features loaded into Pytorch array
											[1]: X_test_tensor: Feature tensors in validation data
											[2]: y_train_tensor: Training labels as Pytorch tensor
											[3]: y_test_tensor: Target labels as tensor in validation data
											[4]: Iterable dataset object and iterating over it in 
												 batches, which are then fed into the model for processing

	Example call:
	---
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	'''

	#################	ADD YOUR CODE HERE	##################
	X_train, X_test, y_train, y_test = train_test_split(
		features_and_targets[0].to_numpy(),
		features_and_targets[1].to_numpy(),
		test_size=0.2,
		random_state=42,
	)

	# converting into tensors for pytorch
	X_train = torch.from_numpy(X_train.astype(numpy.float32))
	X_test = torch.from_numpy(X_test.astype(numpy.float32))
	# unsqueeze is used to add a dimension to the tensor
	y_train = torch.from_numpy(y_train).type(torch.FloatTensor).unsqueeze(1)
	y_test = torch.from_numpy(y_test).type(torch.FloatTensor).unsqueeze(1)

	tensors_and_iterable_training_data = [
		X_train,
		X_test,
		y_train,
		y_test,
		torch.utils.data.DataLoader(
			Data(X_train, y_train), batch_size=4096, shuffle=False
		),
	] # DataLoader is used as the iterable dataset object
	##########################################################

	return tensors_and_iterable_training_data


class Salary_Predictor(torch.nn.Module):
	'''
	Purpose:
	---
	The architecture and behavior of your neural network model will be
	defined within this class that inherits from nn.Module. Here you
	also need to specify how the input data is processed through the layers. 
	It defines the sequence of operations that transform the input data into 
	the predicted output. When an instance of this class is created and data
	is passed through it, the `forward` method is automatically called, and 
	the output is the prediction of the model based on the input data.
	
	Returns:
	---
	`predicted_output` : Predicted output for the given input data
	'''
	def __init__(self):
		super(Salary_Predictor, self).__init__()
		'''
		Define the type and number of layers
		'''
		#######	ADD YOUR CODE HERE	#######
		# input layer
		self.linear1 = torch.nn.Linear(8, 1024)
		# hidden layers
		self.hidden1 = torch.nn.Linear(1024,2048)
		self.hidden2 = torch.nn.Linear(2048,4096)
		self.hidden3 = torch.nn.Linear(4096,2048)
		# output layer
		self.linear2 = torch.nn.Linear(2048, 1)
		###################################

	def forward(self, x):
		'''
		Define the activation functions
		'''
		#######	ADD YOUR CODE HERE	#######
		# relu is utilized as the activation function
		x = torch.relu(self.linear1(x))
		x = torch.relu(self.hidden1(x))
		x = torch.relu(self.hidden2(x))
		x = torch.relu(self.hidden3(x))
		# sigmoid at the end for BCELoss to work
		x = torch.sigmoid(self.linear2(x))
		predicted_output = x
		###################################

		return predicted_output


def model_loss_function():
	'''
	Purpose:
	---
	To define the loss function for the model. Loss function measures 
	how well the predictions of a model match the actual target values 
	in training data.
	
	Input Arguments:
	---
	None

	Returns:
	---
	`loss_function`: This can be a pre-defined loss function in PyTorch
					or can be user-defined

	Example call:
	---
	loss_function = model_loss_function()
	'''
	#################	ADD YOUR CODE HERE	##################
	loss_function = torch.nn.BCELoss()  # for binary classification
	##########################################################

	return loss_function


def model_optimizer(model):
	'''
	Purpose:
	---
	To define the optimizer for the model. Optimizer is responsible 
	for updating the parameters (weights and biases) in a way that 
	minimizes the loss function.
	
	Input Arguments:
	---
	`model`: An object of the 'Salary_Predictor' class

	Returns:
	---
	`optimizer`: Pre-defined optimizer from Pytorch

	Example call:
	---
	optimizer = model_optimizer(model)
	'''
	#################	ADD YOUR CODE HERE	##################
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # Good enough for this model
	##########################################################

	return optimizer

def model_number_of_epochs():
	'''
	Purpose:
	---
	To define the number of epochs for training the model

	Input Arguments:
	---
	None

	Returns:
	---
	`number_of_epochs`: [integer value]

	Example call:
	---
	number_of_epochs = model_number_of_epochs()
	'''
	#################	ADD YOUR CODE HERE	##################
	number_of_epochs = 1000  # found via trial and error
	##########################################################

	return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
	'''
	Purpose:
	---
	All the required parameters for training are passed to this function.

	Input Arguments:
	---
	1. `model`: An object of the 'Salary_Predictor' class
	2. `number_of_epochs`: For training the model
	3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
											 and iterable dataset object of training tensors
	4. `loss_function`: Loss function defined for the model
	5. `optimizer`: Optimizer defined for the model

	Returns:
	---
	trained_model

	Example call:
	---
	trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

	'''
	#################	ADD YOUR CODE HERE	##################
	model.train(True)
	for epoch in range(number_of_epochs):
		running_loss = 0.0

		for i, data in enumerate(tensors_and_iterable_training_data[4], 0):
			inputs, labels = data
			# zero grad for every batch size
			optimizer.zero_grad()

			# training and calculating parameters
			outputs = model(inputs)
			loss = loss_function(outputs, labels)
			loss.backward()

			optimizer.step()

			running_loss += loss.item()

	trained_model = model
	trained_model.train(False)  # ready for eval
	##########################################################

	return trained_model


def validation_function(trained_model, tensors_and_iterable_training_data):
	'''
	Purpose:
	---
	This function will utilise the trained model to do predictions on the
	validation dataset. This will enable us to understand the accuracy of
	the model.

	Input Arguments:
	---
	1. `trained_model`: Returned from the training function
	2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
											 and iterable dataset object of training tensors

	Returns:
	---
	model_accuracy: Accuracy on the validation dataset

	Example call:
	---
	model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

	'''	
	#################	ADD YOUR CODE HERE	##################

	test_data_loader = torch.utils.data.DataLoader(
		Data(
			tensors_and_iterable_training_data[1], tensors_and_iterable_training_data[3]
		),
		batch_size=4096,
		shuffle=False,
	)  # need to load test data using loader like training data
 
	correct, total = 0, 0
	with torch.no_grad():  # no gradient calculation during eval
		for data in test_data_loader:
			inputs, labels = data
			outputs = trained_model(inputs)
			correct += (outputs.round() == labels).sum()  # round() reqd as model output is sigmoid
			total += labels.size(0)

	model_accuracy = (correct / total) * 100
	##########################################################

	return model_accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''
if __name__ == "__main__":
	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	
	# model is an instance of the class that defines the architecture of the model
	model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
					loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")