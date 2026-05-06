from setuptools import setup, find_packages

setup(
    name='ceqnet',
    version='0.1.0',
    description='CeqNet: a machine-learning interatomic potential with self-consistent charge equilibration',
    author='Martin Michajlow',
    packages=find_packages(),
    package_data={'src.sph_ops': ['cgmatrix.npz']},
    python_requires='>=3.9',
    install_requires=[
        "numpy",
        # "jax == 0.4.8",
        "flax",
        "jaxopt",
        "jraph",
        "optax",
        "orbax-checkpoint == 0.5.23",
        "portpicker",
        # 'tensorflow',
        "scikit-learn",
        "ase",
        "tqdm",
        "wandb",
        "pyyaml",
        "h5py",
    ],
)
