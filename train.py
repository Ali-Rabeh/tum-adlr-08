import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from util.dataset import CustomDataset, SequenceDataset
from models.forward_model import ForwardModel

# Largely following the PyTorch quickstart tutorial for now: 
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

hparams =  {
    "batch_size": 16,
    "sequence_length": 2,
    "learning_rate": 1e-10,
    "epochs": 10
}

def assemble_datasets(path_list, mode='sequence', sequence_length=4):
    list_of_datasets = []
    for path in path_list:
        if mode == 'sequence':
            list_of_datasets.append(SequenceDataset(path, sequence_length))
        elif mode == 'regular':
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
    size = len(dataloader.dataset)

    model = model.to(device)
    model.train()

    counter = 0
    for X, y in dataloader:
        counter += 1

        # one step prediction error
        X = X[0:dataloader.batch_size-1,7:10]
        y = y[1:dataloader.batch_size]

        X, y = X.to(device), y.to(device)
        diff_pred = model(X)
        pred = X + diff_pred
        # print(f"Input: {X} | Network Output: {diff_pred} | Full Prediction: {pred} | Target: {y}")

        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if counter % 1 == 0:
            loss, current = loss.item(), (counter+1)*len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    # print(f"Counter: {counter}")

def train_forward_model_simple(model, loss_fn, optimizer, sequence_length):
    model = model.to(device)
    model.train()

    X = torch.tensor([[1., 1., 1.], [1.1, 1.1, 1.1]])
    y = X

    X = X[0:1]
    y = y[1:2]

    X, y = X.to(device), y.to(device)
    diff_pred = model(X)
    pred = X + diff_pred
    print(f"Input: {X} | Network Output: {diff_pred} | Full Prediction: {pred} | Target: {y}")
    print(pred.shape)
    print(y.shape)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    counter, size = 0, 1
    if counter % 10 == 0:
        loss, current = loss.item(), (counter+1)*len(X)
        print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# function for pretraining the forward model
# def train_forward_model(dataloader, model, loss_fn, optimizer, sequence_length): 

#     model.train()

#     # iterate through all batches
#     for batch_counter, data in enumerate(dataloader):
#         running_loss = 0.0
#         last_loss = 0.0
#         loss = torch.zeros(size=[])

#         X = data[0]

#         for sequence_counter in range(sequence_length):
#             # New sequence --> set everything to zero and reinitialize the DPF
#             if sequence_counter % sequence_length == 0:
#                 last_state = None

#             # print(f"i: {sequence_counter}")
#             optimizer.zero_grad()
#             loss = torch.zeros(size=[])

#             object_state = X[:,sequence_counter,7:10]

#             # Predict sequences and get loss
#             if last_state is not None:
#                 delta_particle_states = model(last_state)
#                 estimate = last_state + delta_particle_states

#                 N = object_state.shape[0]
#                 estimate = estimate.reshape(N, -1)
#                 object_state_reshaped = object_state.reshape(N, -1)

#                 # Compute loss
#                 loss += loss_fn(object_state.reshape(N, -1), estimate)
#                 running_loss += loss.item()

#             last_state = object_state # one step prediction error

#             # Do an optimization step after the last sample from the sequence has been predicted
#             if (sequence_counter % sequence_length == sequence_length-1) and (sequence_counter !=0 ):
#                 loss /= sequence_length
#                 loss.backward()
#                 optimizer.step()

#                 last_loss = running_loss / sequence_length 
#                 # print(f"Motion batch {batch_counter+1} loss: {last_loss}")
#                 running_loss = 0.0 # reset the running loss after the sequence

#     return last_loss, model

def test_forward_model(dataloader, model, loss_fn, sequence_length):
    num_batches = len(dataloader)
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X[0:dataloader.batch_size-1,7:10]
            y = y[1:dataloader.batch_size]
            X, y = X.to(device), y.to(device)
            diff_pred = model(X)
            pred = X + diff_pred
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Average test loss: {test_loss:>8f} \n")



# Build the training dataloader
train_path_list = [os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.000.json")] 
                   # os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.698.json")]
training_data = assemble_datasets(train_path_list, mode='regular')
train_dataloader = DataLoader(training_data, batch_size=hparams['batch_size'], shuffle=False, drop_last=True)
print(f"Length training dataloader: {len(train_dataloader)}")

# training_data = assemble_datasets(train_path_list, mode='sequence', sequence_length=hparams['sequence_length'])
# train_dataloader = DataLoader(training_data, batch_size=hparams['batch_size'], shuffle=False)
# print(train_dataloader.dataset)

# Build the testing dataloader
test_path_list = [os.path.abspath("../pushdata/plywood/butter/motion_surface=plywood_shape=butter_a=0_v=10_i=0.000_s=0.000_t=0.349.json")]
test_data = assemble_datasets(test_path_list, mode='regular')
test_dataloader = DataLoader(test_data, batch_size=hparams['batch_size'], shuffle=False, drop_last=True)

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
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=hparams['learning_rate'])

# epochs = hparams['epochs']
# sequence_length = hparams['sequence_length']
# for i in range(epochs):
#     epoch_loss, model = train_forward_model(train_dataloader, model, loss_fn, optimizer, sequence_length)
#     print(f"Epoch {i+1}: loss = {epoch_loss}")
#     test_forward_model(test_dataloader, model, loss_fn, sequence_length)

# Train the model
epochs = hparams['epochs']
for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------------")
    # train_forward_model(train_dataloader, model, loss_fn, optimizer, hparams['sequence_length'])
    train_forward_model_simple(model, loss_fn, optimizer, hparams['sequence_length'])
    # test_forward_model(test_dataloader, model, loss_fn, hparams['sequence_length'])
print("Done.")





