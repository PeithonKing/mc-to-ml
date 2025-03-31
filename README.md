# Learning Ising Model Dynamics with Neural Networks

## About
This project explores the intersection of statistical mechanics and deep learning by training a neural network to simulate the dynamics of the 2D Ising model. Rather than relying solely on traditional Monte Carlo methods, the code implements a model that learns to predict spin-flip probabilities based on local neighborhood configurations and system temperature $T$. This is technically compelling because it attempts to approximate the underlying physical Hamiltonian and replicate critical phenomena, such as the phase transition from a disordered paramagnetic state to an ordered ferromagnetic state, using a learned function.

## Technical Details
The system is built using PyTorch and implements a physics-informed architecture to capture the energetics of spin lattices.

**Mathematical Foundation:**
The model targets the probability of a spin flip, which in a traditional Ising system is governed by the energy difference $\Delta E$. The energy for a spin $s_i$ is defined by:
$$\Delta E = 4 J s_i \sum_{j \in \text{nn}(i)} s_j + 2 h s_i$$
where $J$ is the exchange interaction constant and $h$ is the external magnetic field. The flip probability is typically proportional to $\exp(-\Delta E / T)$.

**Architecture:**
- **Dataset Generation:** `dataset.py` generates synthetic samples of spin configurations and their corresponding flip probabilities based on the energy equation above.
- **HamLearn & Polynomial Model:** To capture the non-linear nature of spin interactions, the architecture uses a `PolynomialModel` paired with circular-padded convolutional layers to simulate periodic boundary conditions.
- **The Predictor:** The main `Model` class pipes the learned Hamiltonian through an exponential activation function $\exp(\cdot)$ and a final MLP, conditioned on the temperature $T$, to output the final flip probability.
- **Simulation Engine:** `utils.py` provides a `generate` function that uses the model as a probability predictor to evolve a lattice over time, allowing for the calculation of macroscopic properties like magnetization $M$ and total energy $E$.

## Execution
The project is structured as a set of modules and Jupyter notebooks for experimentation.

**Prerequisites:**
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- tqdm

**Workflow:**
1. **Training:** Execute `train.ipynb` to train the neural network. This notebook handles the synthetic data generation and optimizes the model using MSE loss to minimize the difference between predicted and theoretical flip probabilities.
2. **Evaluation:** Run `evaluate.ipynb` to perform thermal analysis. This notebook compares the ML model's results against the ground truth (Monte Carlo) by plotting magnetization versus temperature to verify if the model captures the phase transition.
3. **Field Analysis:** Use `h_term.ipynb` to investigate how varying the external field $h$ impacts the magnetization of the system.