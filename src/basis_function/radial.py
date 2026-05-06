# Adapted from mlff (https://github.com/thorben-frank/mlff, commit 99dbf76)
# Original author: Thorben Frank et al.
# Modifications: renamed imports (mlff.src → src); removed BernsteinBasis (unused in CeqNet)

from jax import numpy as jnp
import flax.linen as nn

from src.masking.mask import safe_mask


def get_rbf_fn(key: str) -> type(nn.Module):
    if key == "rbf":
        return RBF
    if key == "phys":
        return PhysNetBasis
    if key == "bessel":
        return BesselBasis


class RBF(nn.Module):
    n_rbf: int
    """
    Number of basis functions.
    """
    r_cut: float
    """
    Cutoff radius.
    """
    r_0: float = 0.

    def setup(self):
        self.offsets = jnp.linspace(self.r_0, self.r_cut, self.n_rbf)  # shape: (n_rbf)
        self.widths = jnp.abs(self.offsets[1] - self.offsets[0])  # shape: (1)
        self.coefficient = 10  # shape: (1)

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Expand molecule_inspect in the RBF basis, used in SchNet
        (see https://proceedings.neurips.cc/paper/2017/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf)

        Args:
            r (Array): Distances, shape: (...,1)

        Returns: The expanded molecule_inspect, shape: (...,n,n,L)
        """
        return jnp.exp(-self.coefficient * (r - self.offsets) ** 2)


class PhysNetBasis(nn.Module):
    n_rbf: int
    """
    Number of basis functions.
    """
    r_cut: float
    """
    Cutoff radius.
    """
    r_0: float = 0.

    def setup(self):
        self.offsets = jnp.linspace(jnp.exp(-self.r_cut), jnp.exp(-self.r_0), self.n_rbf)  # shape: (n_rbf)
        self.coefficient = ((2 / self.n_rbf) * (1 - jnp.exp(-self.r_cut)))**(-2)  # shape: (1)

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Expand molecule_inspect in the basis used in PhysNet (see https://arxiv.org/abs/1902.08408)

        Args:
            r (Array): Distances, shape: (...,1)

        Returns: The expanded molecule_inspect, shape: (...,n,n,L)
        """
        return jnp.exp(-abs(self.coefficient)*(jnp.exp(-r) - self.offsets)**2)


class BesselBasis(nn.Module):
    n_rbf: int
    """
    Number of basis functions.
    """
    r_cut: float
    """
    Cutoff radius. Bessel functions are normalized within the interval [0, r_c] as well as bessel(r_c) = 0, as boundary 
    condition when solving the Helmholtz equation.
    """
    r_0: float = 0.

    def setup(self):
        self.offsets = jnp.arange(0, self.n_rbf, 1)  # shape: (n_rbf)

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Expand molecule_inspect in the Bessel basis (see https://arxiv.org/pdf/2003.03123.pdf)

        Args:
                r (Array): Distances, shape: (...,1)

        Returns: The expanded molecule_inspect, shape: (...,n,n,L)
        """

        f = lambda x: jnp.sin(jnp.pi / self.r_cut * self.offsets * x) / x
        return safe_mask(r != 0, f, r, 0)


class FourierBasis(nn.Module):
    n_rbf: int
    """
    Number of basis functions.
    """
    r_cut: float
    """
    Cutoff radius. Bessel functions are normalized within the interval [0, r_c] as well as bessel(r_c) = 0, as boundary 
    condition when solving the Helmholtz equation.
    """
    r_0: float = 0.

    def setup(self):
        self.offsets = jnp.arange(0, self.n_rbf, 1)  # shape: (n_rbf)

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Expand molecule_inspect in the Bessel basis (see https://arxiv.org/pdf/2003.03123.pdf)

        Args:
                r (Array): Distances, shape: (...,1)

        Returns: The expanded molecule_inspect, shape: (...,n,n,L)
        """

        f = lambda x: jnp.sin(jnp.pi / self.r_cut * self.offsets * x) / x
        return safe_mask(r != 0, f, r, 0)
