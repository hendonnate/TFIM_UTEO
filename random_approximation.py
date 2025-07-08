# this script approximates the probabilities of the state vector based on a random basis vector, time(t), transverse field strength(Gamma), and Ising interaction Strength(V)
import netket as nk
import numpy as np
import itertools #generates all possible bit strings
from scipy.sparse.linalg import eigsh, expm_multiply  # expm_multiply is unused here but fine

N = int(input("Number of qubits (N): "))

hi = nk.hilbert.Spin(s=1/2, N=N)
dim = hi.n_states  # dimensions = 2^N

index = np.random.randint(dim)  # random integer between 0 and dim-1
bitstring = format(index, f'0{N}b')  # convert to binary string with leading zeros

psi0 = np.zeros(dim, dtype=complex) #creates state vector with all 0's
psi0[index] = 1.0  # Start in chosen random basis state

print(f"\nRandom initial basis state |{bitstring}⟩")
print(f"Initial state norm: {np.linalg.norm(psi0):.6f}") #should be 1
print(f"Hilbert space dimension: {dim} (2^{N})")

from netket.operator.spin import sigmax, sigmaz

Gamma = float(input("Gamma: "))
V = float(input("V: "))
t = float(input("Evolution time t: "))

terms = []

for i in range(N):
    P = sigmax(hi, i).to_dense()
    terms.append((Gamma, P))

for i in range(N):
    P = (sigmaz(hi, i) * sigmaz(hi, (i + 1) % N)).to_dense()
    terms.append((V, P))

psi_t = psi0.copy()

for coeff, P in terms:
    theta = coeff * t
    psi_t = np.cos(theta) * psi_t + 1j * np.sin(theta) * (P @ psi_t)

print(f"\nFinal state norm (approximate): {np.linalg.norm(psi_t):.6f}")

basis_states = list(itertools.product([0, 1], repeat=N)) #lists all possible bitstrings with N qubits

print(f"\n{'Basis':>8} {'Amplitude':>30} {'Probability':>15}")
print("-" * 65)

total_prob = 0.0
max_prob = 0.0  # tracks highest probability
max_idx = -1    # index of the basis state with highest probability

for idx, bits in enumerate(basis_states):
    amplitude = psi_t[idx]
    prob = np.abs(amplitude)**2
    total_prob += prob

    if prob > max_prob:
        max_prob = prob
        max_idx = idx

    bits_str = ''.join(map(str, bits))
    print(f"|{bits_str}⟩ {amplitude:>30.6f} {prob:15.6f}")

print("-" * 65)
print(f"Total probability: {total_prob:.6f}")

most_probable_bits = ''.join(map(str, basis_states[max_idx]))
print(f"\nMost probable basis state: |{most_probable_bits}⟩ with probability {max_prob:.6f}")
