# this script approximates the probabilities of the state vector based on an inputted basis vector, time(t), transverse field strength(Gamma), and Ising interaction Strength(V)
import netket as nk
import numpy as np
import itertools

N = int(input("Number of qubits (N): "))

hi = nk.hilbert.Spin(s=1/2, N=N)
dim = hi.n_states

bitstring = input(f"Enter initial basis state as {N} bits (e.g., 100 or 010): ").strip()
if len(bitstring) != N or any(c not in '01' for c in bitstring):
    raise ValueError(f"Invalid input. It must be {N} bits of 0 or 1.")

index = int(bitstring, 2)

psi0 = np.zeros(dim, dtype=complex)
psi0[index] = 1.0

print(f"\nInitial basis state |{bitstring}⟩")
print(f"Initial state norm: {np.linalg.norm(psi0):.6f}")
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
print(f"\nMost probable basis state: |{most_probable_bits}⟩ with probability {max_prob:.6f}")
