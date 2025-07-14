import netket as nk
import numpy as np
import itertools
from scipy.sparse.linalg import eigsh, expm_multiply
Lx = int(input("Lattice size x: "))
Ly = int(input("Lattice size y: "))
N = Lx * Ly
hi = nk.hilbert.Spin(s=1/2, N=N)
dim = hi.n_states  # 2^(Lx*Ly)
index = np.random.randint(dim)
bitstring = format(index, f'0{N}b')
psi0 = np.zeros(dim, dtype=complex)
psi0[index] = 1.0
print(f"\nRandom initial basis state |{bitstring}⟩")
print(f"Initial state norm: {np.linalg.norm(psi0):.6f}")
print(f"Hilbert space dimension: {dim} (2^{N})")
from netket.operator.spin import sigmax, sigmaz
Gamma = float(input("Gamma: "))
V = float(input("V: "))
H = 0
def site(i, j):
    return i * Ly + j
for i in range(Lx):
    for j in range(Ly):
        idx = site(i, j)
        H += Gamma * sigmax(hi, idx)
for i in range(Lx):
    for j in range(Ly):
        idx = site(i, j)
        # Right neighbor
        if j + 1 < Ly:
            right = site(i, j+1)
            H += V * sigmaz(hi, idx) * sigmaz(hi, right)
        # Down neighbor
        if i + 1 < Lx:
            down = site(i+1, j)
            H += V * sigmaz(hi, idx) * sigmaz(hi, down)
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
print(f"\nMost probable basis state: |{most_probable_bits}⟩ with probability {max_prob:.6f}")
