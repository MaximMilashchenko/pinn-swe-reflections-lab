import numpy as np
import torch
from smt.sampling_methods import LHS
from torch.utils.data import DataLoader, Dataset


class PDEDataset(Dataset):
    def __init__(self, minimum_time, maximum_time, minimum_x, maximum_x, batch_size):
        self.minimum_time = minimum_time
        self.maximum_time = maximum_time
        self.minimum_x = minimum_x
        self.maximum_x = maximum_x
        self.symbolic_function_batch_size = batch_size

        self.symbolic_function_sampling_method = LHS(
            xlimits=np.array([[self.minimum_time, self.maximum_time], [self.minimum_x, self.maximum_x]])
        )

        self.tx = torch.FloatTensor(
            self.symbolic_function_sampling_method(self.symbolic_function_batch_size)
        ).requires_grad_(True)
        self.t = self.tx[:, 0:1]
        self.x = self.tx[:, 1:2]
        self.n_samples = self.tx.shape[0]

    def __getitem__(self, index):
        return self.tx[index, :]

    def __len__(self):
        return self.n_samples


class LowerBoundaryDataset(Dataset):
    def __init__(self, minimum_time, maximum_time, minimum_x, batch_size):
        self.minimum_time = minimum_time
        self.maximum_time = maximum_time
        self.minimum_x = minimum_x
        self.lower_boundary_batch_size = batch_size

        self.lower_boundary_sampling_method = LHS(
            xlimits=np.array([[self.minimum_time, self.maximum_time], [self.minimum_x, self.minimum_x]])
        )

        self.tx = torch.FloatTensor(self.lower_boundary_sampling_method(self.lower_boundary_batch_size)).requires_grad_(
            True
        )
        self.t = self.tx[:, 0:1]
        self.x = self.tx[:, 1:2]
        self.n_samples = self.tx.shape[0]

    def __getitem__(self, index):
        return self.tx[index, :]

    def __len__(self):
        return self.n_samples


class UpperBoundaryDataset(Dataset):
    def __init__(self, minimum_time, maximum_time, maximum_x, batch_size):
        self.minimum_time = minimum_time
        self.maximum_time = maximum_time
        self.maximum_x = maximum_x
        self.upper_boundary_batch_size = batch_size

        self.upper_boundary_sampling_method = LHS(
            xlimits=np.array([[self.minimum_time, self.maximum_time], [self.maximum_x, self.maximum_x]])
        )

        self.tx = torch.FloatTensor(self.upper_boundary_sampling_method(self.upper_boundary_batch_size)).requires_grad_(
            True
        )
        self.t = self.tx[:, 0:1]
        self.x = self.tx[:, 1:2]
        self.n_samples = self.tx.shape[0]

    def __getitem__(self, index):
        return self.tx[index, :]

    def __len__(self):
        return self.n_samples


class InitialConditionDataset(Dataset):
    def __init__(self, minimum_time, minimum_x, maximum_x, batch_size):
        self.minimum_time = minimum_time
        self.minimum_x = minimum_x
        self.maximum_x = maximum_x
        self.initial_condition_batch_size = batch_size

        self.initial_condition_sampling_method = LHS(
            xlimits=np.array([[self.minimum_time, self.minimum_time], [self.minimum_x, self.maximum_x]])
        )

        self.tx = torch.FloatTensor(
            self.initial_condition_sampling_method(self.initial_condition_batch_size)
        ).requires_grad_(True)
        self.t = self.tx[:, 0:1]
        self.x = self.tx[:, 1:2]
        self.n_samples = self.tx.shape[0]

    def __getitem__(self, index):
        return self.tx[index, :]

    def __len__(self):
        return self.n_samples


class FinalStateDataset(Dataset):
    def __init__(self, maximum_time, minimum_x, maximum_x, batch_size):
        self.maximum_time = maximum_time
        self.minimum_x = minimum_x
        self.maximum_x = maximum_x
        self.initial_condition_batch_size = batch_size

        self.final_state_sampling_method = LHS(
            xlimits=np.array([[self.maximum_time, self.maximum_time], [self.minimum_x, self.maximum_x]])
        )

        self.tx = torch.FloatTensor(self.final_state_sampling_method(self.initial_condition_batch_size)).requires_grad_(
            True
        )
        self.t = self.tx[:, 0:1]
        self.x = self.tx[:, 1:2]
        self.n_samples = self.tx.shape[0]

    def __getitem__(self, index):
        return self.tx[index, :]

    def __len__(self):
        return self.n_samples


dataset = PDEDataset(minimum_time=0.0, maximum_time=129000.0, minimum_x=-1000000, maximum_x=1000000, batch_size=50000)
batch_size = 1000
n_epochs = 10
n_iterations = int(dataset.__len__() / batch_size)
for i in range(n_epochs):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    dataiter = iter(dataloader)
    for j in range(n_iterations):
        next(dataiter)
