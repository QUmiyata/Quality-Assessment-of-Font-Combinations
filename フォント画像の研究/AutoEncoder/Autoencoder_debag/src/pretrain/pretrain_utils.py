import torch
import numpy as np
import re

import os
import sys
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib'))
# sys.path.append(parent_dir)


class EarlyStopping:
    '''
    validation lossがpatience回以上更新されなければself.early_stopをTrueに
    '''
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.min_value = np.Inf
    def __call__(self, value):
        if value >= self.min_value-self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            print(f'Validation loss decreased ({self.min_value} --> {value})')
            self.min_value = value


def train(dataloader, model, criterion, optimizer, device):
    '''Train the model for one epoch and calculate average loss.'''
    model.train()
    epoch_loss = 0.0
    num_samples = 0

    all_latent = []
    all_inputs = []
    all_outputs = []

    for data in dataloader:
        # Forward pass and loss computation
        input_imgs = data.to(device)

        latent, output_imgs = model(input_imgs)

        all_latent.append(latent)
        all_inputs.append(input_imgs)
        all_outputs.append(output_imgs)

        loss = criterion(output_imgs, input_imgs)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss and count samples
        batch_size_actual = data.size(0)
        epoch_loss += loss.item() * batch_size_actual
        num_samples += batch_size_actual

    all_latent_epoch = torch.cat(all_latent, dim=0)
    all_inputs_epoch = torch.cat(all_inputs, dim=0)
    all_outputs_epoch = torch.cat(all_outputs, dim=0)
    # Return average loss for the epoch
    return epoch_loss / num_samples, all_latent_epoch, all_inputs_epoch, all_outputs_epoch


def val(dataloader, model, criterion, device):
    '''Validate the model and calculate average loss.'''
    model.eval()
    epoch_loss = 0.0
    num_samples = 0

    all_latent = []
    all_inputs = []

    with torch.no_grad():
        for data in dataloader:
            # Forward pass and loss computation
            input_imgs = data.to(device)
            latent, output_imgs = model(input_imgs)

            all_latent.append(latent)
            all_inputs.append(input_imgs)

            loss = criterion(output_imgs, input_imgs)

            # Accumulate loss and count samples
            batch_size = input_imgs.size(0)
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

        all_latent_epoch = torch.cat(all_latent, dim=0)
        all_inputs_epoch = torch.cat(all_inputs, dim=0)

    # Return average loss for the validation dataset
    return epoch_loss / num_samples, all_latent_epoch, all_inputs_epoch
