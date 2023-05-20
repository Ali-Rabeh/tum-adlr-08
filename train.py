import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from util.dataset import CustomDataset
from models.forward_model import ForwardModel

# Largely following the PyTorch quickstart tutorial for now: 
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

def assemble_datasets(path_list):
    list_of_datasets = []
    for path in path_list:
        list_of_datasets.append(CustomDataset(path))
    return ConcatDataset(list_of_datasets)

# standard train function from pytorch tutorials
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# standard test function from pytorch tutorials
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred_argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches 
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg. loss: {test_loss:>8f} \n")

def train_forward_model(dataloader, model, loss_fn, optimizer, sequence_length): 

    running_loss = 0.0
    last_loss = 0.0
    loss = torch.zeros(size=[])

    for i, data in enumerate(dataloader):
        # New sequence --> set everything to zero and reinitialize the DPF
        if i % sequence_length == 0:
            last_state = None

        optimizer.zero_grad()
        loss = torch.zeros(size=[])

        object_state = data[1]

        # Predict sequences and get loss
        if last_state is not None:
            delta_particle_states = model(last_state)
            estimate = last_state + delta_particle_states

            # Why is that all necessary? 
            N = object_state.shape[0]
            estimate = estimate.reshape(N, -1)
            object_state_reshaped = object_state.reshape(N, -1)

            # Compute loss
            loss += loss_fn(object_state.reshape(N, -1), estimate)
            running_loss += loss.item()

        last_state = object_state

        # Do an optimization step after the last sample from the sequence has been predicted
        if (i % sequence_length == sequence_length-1) and (i !=0 ):
            loss /= sequence_length
            loss.backward()
            optimizer.step()

            last_loss = running_loss / (sequence_length * sequence_length) 
            # print(f"Motion batch {i+1} loss: {last_loss}")
            running_loss = 0.0 # reset the running loss after the sequence

    return last_loss, model



# Build the training dataloader
train_path_list = [os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.000.json"), 
             os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.349.json")]
training_data = assemble_datasets(train_path_list)
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)
print(train_dataloader.dataset)

# Build the testing dataloader
test_path_list = [os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.698.json")]
test_data = assemble_datasets(test_path_list)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

# Verify the device we are using
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Initialize the model
input_dimension = 3
state_dimension = 3
model = ForwardModel(input_dimension, state_dimension)
model = model.to(device)
print(model)

# Verify the dataloader
for X, y in train_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    break

# Set loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
sequence_length = 4
for i in range(epochs):
    epoch_loss, model = train_forward_model(train_dataloader, model, loss_fn, optimizer, sequence_length)
    print(f"Epoch {i+1}: loss = {epoch_loss}")



# Some tests
# print(f"Model input: {X[0:16,7:10]}")
# delta = model(X[0:16,7:10])
# print(f"Model output: {out}")

# Train the model
# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-----------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done.")





