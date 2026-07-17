# AMP QM/MM

PyTorch implementation for ML/MM with anisotropic message passing (AMP).

# Publications

```bibtex
@article{AMP2,
  title = {Neural Network Potential with Multiresolution Approach Enables Accurate Prediction of Reaction Free Energies in Solution},
  volume = {147},
  ISSN = {1520-5126},
  url = {http://dx.doi.org/10.1021/jacs.4c17015},
  DOI = {10.1021/jacs.4c17015},
  number = {8},
  journal = {J. Am. Chem. Soc.},
  publisher = {American Chemical Society (ACS)},
  author = {Pultar,  Felix and Th\"{u}rlemann,  Moritz and Gordiy,  Igor and Doloszeski,  Eva and Riniker,  Sereina},
  year = {2025},
  pages = {6835--6856}
}
```

# Abstract

We present design and implementation of a novel neural network potential (NNP) and its combination with an electrostatic embedding scheme, commonly used within the context of hybrid quantum-mechanical/molecular-mechanical (QM/MM) simulations. Substitution of a computationally expensive QM Hamiltonian by a NNP with the same accuracy largely reduces the computational cost and enables efficient sampling in prospective MD simulations, the main limitation faced by traditional QM/MM set-ups. The model relies on the recently introduced anisotropic message passing (AMP) formalism to compute atomic interactions and encode symmetries found in QM systems. AMP is shown to be highly efficient in terms of both data and computational costs, and can be readily scaled to sample systems involving more than 350 solute and 40'000 solvent atoms for hundreds of nanoseconds using umbrella sampling. The performance and broad applicability of our approach are showcased by calculating the free-energy surface of alanine dipeptide, the preferred ligation states of nickel phosphine complexes, and dissociation free energies of charged pyridine and quinoline dimers. Results with this ML/MM approach show excellent agreement with experimental data. In contrast, free energies calculated with static high-level QM calculations paired with implicit solvent models or QM/MM MD simulations using cheaper semi-empirical methods show up to ten times higher deviation from the experimental ground truth and sometimes even fail to reproduce qualitative trends.

# Data

- Datasets to train AMP have been made available via [ETH Research Collection](https://dx.doi.org/10.3929/ethz-b-000707814)

Note that these datasets are provided with maximum storage efficiency (hdf5) and have to be pre-processed to yield a format that can be consumed by AMP (npy). An example is provided under the same link.

# Installation 

Installation has been tested on CPU and NVIDIA GPU hardware. While CPU installations are much easier, CUDA is recommended for best performance.

## AMP - Python

Assuming a `conda` environment with a PyTorch installation is present, the repository can be installed with:

```bash
git clone --recursive https://github.com/rinikerlab/amp_qmmm --depth 1 
conda install torchmetrics pytorch-scatter pytorch-sparse pytorch-cluster -c pyg -c conda-forge
pip install torchlayers tensorboard 
```

Please refer to the PyTorch and `conda` documentation as well as to the PyTorch Geometric website for installation instructions and comptatible versions. For example, the following combination of libraries has been tested successfully on Ubuntu 22.04:

```bash
conda install python=3.11 pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 numpy=1.26.4 torchmetrics pytorch-scatter=2.1.2  pytorch-sparse=0.6.18 pytorch-cluster=1.6.3 -c pytorch -c nvidia -c pyg -c conda-forge
pip install torchlayers tensorboard 
```

`AMPQMMM.py` is supposed to be used for larger models (2.7M parameters) generally not recommended. `AMPQMMMmin.py` is supposed to be used instead (600k parameters). Refer to the publication for a discussion. 

## GROMOS with PyTorch Plugin - C++

A branch of GROMOS interfaced to `libtorch` is provided [here](https://github.com/rinikerlab/gromosXX) (check out branch: `torch`). In order to build this branch, the `cmake` build system has to know the location of `libtorch`, `libxtb`, and their dependencies in addition to other libraries GROMOS depends on. In its basic form, this can be achieved via:

```
cmake -S . -B build -DMPI=on -DXTB=on -DTORCH=on 
cmake --build build 
cmake --install build
```

These commands will build a version of GROMOS interfaced to `xtb` and `libtorch` (potentially with CUDA support), parallelized with MPI for solvent/solvent interactions. Note that the environment variables `CMAKE_PREFIX_PATH`, `PKG_CONFIG_PATH`, `LIBRARY_PATH`, and `LD_LIBRARY_PATH` have to be set appropiately for `cmake` to locate dependencies. A self-documenting `Dockerfile` is provided in the `docker` subdirectory. Make sure, files in the `patch` folder are used, which will update the C++ version to 17 (required by PyTorch) and enable selection of an implicit solvent model for `xtb`.

## Docker Image

Due to the more involved installation process, a containerized version of GROMOS interfacing `xtb` and `PyTorch` with CUDA is provided. The image also contains analysis tools GROMOS++ and AMP training and test scripts to enable the full pipeline from training a model to a prospective MD simulation. The version of CUDA and PyTorch supported by the GPU used may vary. Consult your GPU documentation and replace the `GPU_ARCHITECTURE` variable accordingly. The Dockerfile can also serve as template for the installation described above.

### Installation and Usage

Make sure to follow instructions provided by Docker (and NVIDIA) to set up Docker with support for CUDA.

### Build the Image

From the current folder, run

```bash
docker build -t amp:pytorch-2.2.2-cuda12.1  .
```
Passing the option `--ulimit nofile=100000:100000` may help if the build fails due to too many open files.

### Run the Image

```bash
docker run -v ./data:/workspace/amp_qmmm/data -v ./inputs:/workspace/amp_qmmm/inputs -v ./results:/workspace/amp_qmmm/results -v ./summaries:/workspace/amp_qmmm/summaries  -v ./examples:/workspace/examples --gpus '"device=0"' --rm -it --ipc=host amp:pytorch-2.2.2-cuda12.1
```

The following folders are expected to be present in the current directory and are mapped to the container for I/O (see docker folder):

- data: training and test data
- inputs: hyperparameter configurations
- results: model weights and scripted models after training
- summaries: for TensorBoard

# Usage

## Training

To train a model, create a yaml file that contains the configuration. The subdirectory `scripts` contains code to create a template. For recommendations on specific hyperparamter combinations, refer to the original publication. Pre-process QM reference data in analogy to the example shown in the `docker` subdirectory. Afterwards, training can be started with:

```bash
python train_amp.py parameters.yaml
```

## Testing

During training, a folder with results is generated, which contains model weights and initial hyperparameters. To collect test metrics on a trained model, run:

```bash
python test_amp.py results_folder
```

## Conversion

Scripting is used to have the model run in the C++ layer of GROMOS:

```bash
python convert_amp.py results_folder
```

Choose the `.pt` file, which corresponds to the platform (CPU vs CUDA) and data type (float32 vs float64).

## Simulation

ML/MM MD simulations with GROMOS are set up as any other QM/MM simulation with the following exceptions:

(1) The PyTorch environment has to be activated in the `.imd` file:

```
TORCH
# TORCH
       1
END
```

(2) A `.torch` specification file has to be created, which contains details on the model used:

```
TITLE
Torch Specification File
END
MODELS
# TORNAM: Name of the Torch model
# TORATM: 0..3 Atom selection scheme
#    0: All atoms to Torch
#    1: QM zone to Torch
#    2: Custom selection scheme
# TORFIL: Filename of the serialized model
# TORPRC: Numerical minimal of the model
#    0: float16
#    1: float32
#    2: float64
# TORDEV: Device the model runs on
#    0: autodetect
#    1: CPU
#    2: CUDA
# TORLEN: Conversion factor to convert the ML length unit to the GROMOS one
# TORENE: Conversion factor to convert the ML energy unit to the GROMOS one
# TORFOR: Conversion factor to convert the ML force unit to the GROMOS one
# TORCHR: Conversion factor to convert the ML charge unit to the GROMOS one
# TORNAM                  TORATM                         TORFIL        TORPRC  TORDEV  TORWRT  TORLE   TORENE   TORFOR   TORCH
  model_em_sys-4ba-01_1        1          model_float32_cuda.pt        1       2       500     0.1     1.0       10      1
END
```

For AMP models, `TORATM` is set to `1` (QM zone to Torch), `float32` is recommended, and Angstrom is used (GROMOS uses nm internally).

(3) A `.qmmm` specification file has to be created, according to the GROMOS documentation, which specifies electrostatic embedding and the `Ghost worker` (which gathers QM/MM atoms and sends them to PyTorch):

```
QMMM
#  NTQMMM     NTQMSW     RCUTQ          NTWQMMM      QMLJ    QMCON     MMSCAL
   2          -1          1.4            500            0       0        -1.0
END
```

An example is provided in the `examples` subdirectory.

# Authors

Felix Pultar ([@pultar](https://github.com/pultar)), Moritz Thürlemann ([@MOSNPDEV](https://github.com/MOSNPDEV))
