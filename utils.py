import torch
import torch.nn.functional as F
from dataset import K
import numpy as np
from tqdm import tqdm, trange

# def generate(predictor, res, T, n=100, p=(1.0, 0.1), in_channels=1):
#     B = len(T)
#     lattice = torch.where(torch.rand(B, in_channels, res, res) < 0.5, 1.0, -1.0)
#     ps = torch.linspace(p[0], p[1], n)
#     with torch.no_grad():
#         for i in range(n):
#             prob = predictor(lattice, T) * ps[i]
#             mask = torch.rand_like(prob) < prob
#             lattice[mask] *= -1
#     return lattice

def generate(predictor, res, T, n=100, p=(1.0, 0.1), in_channels=1):
    B = len(T)
    lattice = torch.where(torch.rand(B, in_channels, res, res) < 0.5, 1.0, -1.0)
    ps = torch.linspace(p[0], p[1], n)
    
    with torch.no_grad():
        for i in range(n):
            prob = predictor(lattice, T) * ps[i]
            mask = torch.rand_like(prob) < prob
            lattice[mask] *= -1
        
        plus_count = (lattice == 1.0).sum()
        minus_count = (lattice == -1.0).sum()
        majority_value = 1.0 if plus_count >= minus_count else -1.0
        a = 0.5
        for i in range(n // 2, n):
            prob = predictor(lattice, T) * ps[i]

            # Adjust probabilities based on majority/minority status
            adjustment = torch.where(lattice == majority_value, 
                                    prob * (1-a),  # Majority: 0.1 times less probability
                                    prob * (1+a))  # Minority: 0.1 times more probability
            
            mask = torch.rand_like(adjustment) < adjustment
            lattice[mask] *= -1
        
        for i in range(n):
            prob = predictor(lattice, T) * ps[i]
            mask = torch.rand_like(prob) < prob
            lattice[mask] *= -1
            
    return lattice

# def f(lattice, T, J):
#     sj = F.conv2d(
#         F.pad(lattice, (1, 1, 1, 1), "circular"),
#         K, padding=0,
#     )
#     Jsisj = -J*lattice*sj
#     neg_deltaE = 4*Jsisj
#     prob = torch.exp(neg_deltaE/T.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
#     return torch.clamp(prob, min=0, max=1)

def f(lattice, T, J, h=0):
    sj = F.conv2d(
        F.pad(lattice, (1, 1, 1, 1), "circular"),
        K, padding=0,
    )
    Jsisj = -J * lattice * sj
    h_term = -h * lattice  # External field contribution
    neg_deltaE = 4 * Jsisj + 2 * h_term
    prob = torch.exp(neg_deltaE / T.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
    return torch.clamp(prob, min=0, max=1)

def get_magnetizations(lattices):
    return lattices.mean(dim=[1, 2, 3])

def get_energies(lattices, J=1.0):
    sj = F.conv2d(
        F.pad(lattices, (1, 1, 1, 1), "circular"),
        K, padding=0,
    )
    Jsisj = -J * lattices * sj
    return Jsisj.sum(dim=[1, 2, 3])

def thermal_analysis(predictor, J, res=32, n=100, p=(1.0, 0.1), in_channels=1):
    temperatures = np.arange(1, 20, 0.5)
    avg_energies = np.zeros(len(temperatures))
    avg_magnetizations = np.zeros(len(temperatures))
    
    for i, temp in enumerate(tqdm(temperatures)):
        T = torch.ones(100) * temp
        lattices = generate(predictor, res, T, n, p, in_channels)
        energies = get_energies(lattices, J)
        magnetizations = get_magnetizations(lattices)
        abs_magnetizations = torch.abs(magnetizations)
        avg_energies[i] = energies.mean().item()
        avg_magnetizations[i] = abs_magnetizations.mean().item()
    return temperatures, avg_energies, avg_magnetizations

def thermal_analysis_dist(predictor, J, res=32, n=100, p=(1.0, 0.1), in_channels=1):
    temperatures = np.arange(2, 7, 0.1)
    all_energies = []
    all_magnetizations = []
    
    for i, temp in enumerate(tqdm(temperatures)):
        T = torch.ones(100) * temp
        lattices = generate(predictor, res, T, n, p, in_channels)
        energies = get_energies(lattices, J)
        magnetizations = get_magnetizations(lattices)
        all_energies.append(energies)
        all_magnetizations.append(magnetizations)
    return temperatures, np.array(all_energies), np.array(all_magnetizations)

def initialize_parameters(model, freeze=True):  # this is cheating and for testing purposes only
    with torch.no_grad():
        model.conv.conv.weight[0, 0] = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).float()
        model.conv.conv.weight[1, 0] = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).float()
        model.conv.conv.bias *= 0

        # model.conv.poly.conv_layers[0].weight[0] = torch.tensor([1, 0]).float().unsqueeze(-1).unsqueeze(-1)
        # model.conv.poly.conv_layers[1].weight[0] = torch.tensor([0, 1]).float().unsqueeze(-1).unsqueeze(-1)
        # model.conv.poly.conv_layers[0].bias *= 0
        # model.conv.poly.conv_layers[1].bias *= 0
        
        # model.lin1.weight[0] = torch.tensor([[[4.0]]])
        # model.lin1.bias *= 0

    if freeze:
        model.conv.conv.weight.requires_grad = False
        model.conv.conv.bias.requires_grad = False
        # model.conv.poly.conv_layers[0].weight.requires_grad = False
        # model.conv.poly.conv_layers[1].weight.requires_grad = False
        # model.conv.poly.conv_layers[0].bias.requires_grad = False
        # model.conv.poly.conv_layers[1].bias.requires_grad = False
        # model.lin1.weight.requires_grad = False
        # model.lin1.bias.requires_grad = False
        
    return model