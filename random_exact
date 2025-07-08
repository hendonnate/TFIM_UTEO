# this script calculates the probabilities of the state vector based on a random basis vector, time(t), transverse field strength(Gamma), and Ising interaction Strength(V)
import netket as nk
import jax
import numpy as np
import itertools
from scipy.sparse.linalg import eigsh, expm_multiply

N = int(input("Number of qubits (N): "))

hi = nk.hilbert.Spin(s=1/2, N=N)
dim = hi.n_states

psi0 = np.random.randn(dim) + 1j * np.random.randn(dim)
psi0 /= np.linalg.norm(psi0)

print(f"\nRandom initial state generated.")
print(f"Initial state norm: {np.linalg.norm(psi0):.6f}")
print(f"Hilbert space dimension: {dim} (2^{N})")

from netket.operator.spin import sigmax, sigmaz

Gamma = float(input("Gamma: "))
V = float(input("V: "))

H = sum([Gamma * sigmax(hi, i) for i in range(N)])
H += sum([V * sigmaz(hi, i) * sigmaz(hi, (i + 1) % N) for i in range(N)])

sp_h = H.to_sparse()

eig_vals, _ = eigsh(sp_h, k=2, which="SA")
print("\nLowest eigenvalues:", eig_vals)

t = float(input("Evolution time t: "))
psi_t = expm_multiply(1j * sp_h * t, psi0)

print(f"\nFinal state norm: {np.linalg.norm(psi_t):.6f}")

basis_states = list(itertools.product([0, 1], repeat=N))

print(f"\n{'Basis':>8} {'Amplitude':>30} {'Probability':>15}")
print("-" * 65)

total_prob = 0.0
max_prob = 0.0
max_idx = -1

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
print(f"\nMost likely state: |{most_probable_bits}⟩ with probability {max_prob:.6f}")
