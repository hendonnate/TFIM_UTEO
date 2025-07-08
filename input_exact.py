# this script calculates the probabilities of the state vector based on an inputted basis vector, time(t), transverse field strength(Gamma), and Ising interaction Strength(V)
import netket as nk
import jax
import numpy as np
import itertools #generates all possible bit strings
from scipy.sparse.linalg import eigsh, expm_multiply

N = int(input("Number of qubits (N): "))

hi = nk.hilbert.Spin(s=1/2, N=N)
dim = hi.n_states  # dimensions = 2^N

bitstring = input(f"Enter initial basis state as {N} bits ,something like 100 or 010 for N = 3: ").strip()
if len(bitstring) != N or any(c not in '01' for c in bitstring):
    raise ValueError(f"Invalid input. It must be {N} bits of 0 or 1.")

index = int(bitstring, 2)

psi0 = np.zeros(dim, dtype=complex) #creates state vector with all 0's
psi0[index] = 1.0  # Start in chosen basis state

print(f"\nInitial basis state |{bitstring}⟩")
print(f"Initial state norm: {np.linalg.norm(psi0):.6f}") #prints state norm to 6decimal points, should be 1
print(f"Hilbert space dimension: {dim} (2^{N})")

from netket.operator.spin import sigmax, sigmaz

Gamma = float(input("Gamma: "))
V = float(input("V: "))

H = sum([Gamma * sigmax(hi, i) for i in range(N)])
H += sum([V * sigmaz(hi, i) * sigmaz(hi, (i + 1) % N) for i in range(N)])

sp_h = H.to_sparse()
#builds TFIM and converts it to sparse matrix, solving for H in equation

eig_vals, _ = eigsh(sp_h, k=2, which="SA")
print("\nLowest eigenvalues:", eig_vals)
#computes and prints lowest smallest algebraic eigenvalues

t = float(input("Evolution time t: "))
psi_t = expm_multiply(1j * sp_h * t, psi0) #1j refers to sqrt(-1)

print(f"\nFinal state norm: {np.linalg.norm(psi_t):.6f}") #should still be 1

basis_states = list(itertools.product([0, 1], repeat=N)) #lists all possible bitstrings with N qubits

print(f"\n{'Basis':>8} {'Amplitude':>30} {'Probability':>15}")
print("-" * 65)
#sets up the output table, the numbers refer to character length

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
# adds up total probability and tracks highest

print("-" * 65)
print(f"Total probability: {total_prob:.6f}")

# print most probable basis state
most_probable_bits = ''.join(map(str, basis_states[max_idx]))
print(f"\nMost likely state: |{most_probable_bits}⟩ with probability {max_prob:.6f}")
