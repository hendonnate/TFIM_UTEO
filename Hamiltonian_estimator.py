# this script estimates the value of H for a large system of qubits (about n > 17)
import netket as nk
import jax
import jax.numpy as jnp #numpy on roids
from flax import linen as nn #neural network API
from flax import nnx
import netket.nn as nknn #what symm is ran on


N = int(input("Number of Qubits: "))
hi = nk.hilbert.Spin(s=1 / 2, N=N) #hilbert space(the complete complex vector space, with dimensions = 2^N)

Gamma = float(input("Gamma: "))
V = float(input("V: "))

from netket.operator.spin import sigmax, sigmaz

graph = nk.graph.Chain(length=N, pbc=True)

H = sum([Gamma * sigmax(hi, i) for i in range(N)])
H += sum([V * sigmaz(hi, i) * sigmaz(hi, j) for (i, j) in graph.edges()])


class SymmModel(nnx.Module):
    def __init__(self, N: int, alpha: int = 4, *, rngs: nnx.Rngs):
        self.alpha = alpha
        dense_symm_linen = nknn.DenseSymm(
            symmetries=graph.translation_group(),
            features=alpha,
            kernel_init=nn.initializers.normal(stddev=0.01),
        )
        self.linear_symm = nnx.bridge.ToNNX(dense_symm_linen, rngs=rngs).lazy_init(
            jnp.ones((1, 1, N))
        )

    def __call__(self, x: jax.Array):
        x = x.reshape(-1, 1, x.shape[-1])  # (batch, 1, N)
        x = self.linear_symm(x)
        x = nnx.relu(x)
        return jnp.sum(x, axis=(-1, -2))

sampler = nk.sampler.MetropolisLocal(hi)
model = SymmModel(N=N, alpha=4, rngs=nnx.Rngs(0))
vstate = nk.vqs.MCState(sampler, model, n_samples=1024)

optimizer = nk.optimizer.Sgd(learning_rate=0.1)

gs = nk.driver.VMC(
    H,
    optimizer,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1),
)

log = nk.logging.RuntimeLog()
gs.run(n_iter=600, out=log)

symm_energy = vstate.expect(H)
print(f"Final optimized energy: {symm_energy.mean}")
