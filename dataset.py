import torch
import random
from torch.utils.data import Dataset
from torch.nn import functional as F

K = torch.tensor([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
]).float().unsqueeze(0).unsqueeze(0)

# class CustomDataset(Dataset):
#     def __init__(
#         self,
#         num_samples,
#         image_size,
#         channels=1,
#         J=1,
#     ):
#         self.num_samples = num_samples
#         self.image_size = image_size
#         self.channels = channels
#         self.J = J

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         T = 2 + random.random()*5
#         input_image = torch.where(torch.rand(self.channels, self.image_size, self.image_size)<random.random(), 1.0, -1.0)
#         sj = F.conv2d(
#             F.pad(input_image, (1, 1, 1, 1), "circular"),
#             K, padding=0,
#         )
#         Jsisj = self.J*input_image*sj
        
#         # return input_image, -4*Jsisj/T, T
        
#         deltaE = 4*Jsisj
#         prob = torch.exp(-deltaE/T)
#         return input_image, torch.clamp(prob, min=0, max=1), T


class CustomDataset(Dataset):
    def __init__(
        self,
        num_samples,
        image_size,
        channels=1,
        J=1,
        h=0,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.channels = channels
        self.J = J
        self.h = h

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = 2 + random.random() * 5
        input_image = torch.where(
            torch.rand(self.channels, self.image_size, self.image_size) < random.random(),
            1.0,
            -1.0,
        )
        sj = F.conv2d(
            F.pad(input_image, (1, 1, 1, 1), "circular"),
            K,
            padding=0,
        )
        Jsisj = self.J * input_image * sj
        h_term = self.h * input_image

        deltaE = 4 * Jsisj + 2 * h_term  # Including external field effect
        prob = torch.exp(-deltaE / T)
        return input_image, torch.clamp(prob, min=0, max=1), T
