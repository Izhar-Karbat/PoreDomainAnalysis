# Advanced Installation for Pocket Analysis Module

The `pocket_analysis` module in the Pore Analysis Suite relies on machine learning libraries (`torch`, `torch_geometric`, `torchmd-net`) and performs best with a CUDA-enabled GPU. Installing these dependencies, especially `torchmd-net` with CUDA support, requires specific steps.

This guide provides instructions based on **Ubuntu 24.04 with CUDA 12.4**. Adapt commands as necessary for your operating system and CUDA version. Using a Conda/Mamba environment is strongly recommended.

---

## 🧱 Step 1: Install System Dependencies (Example for Ubuntu)

```bash
sudo apt update
sudo apt install -y build-essential gcc g++ # Ensure build tools are present (gcc/g++ version might need adjustment)
# You might need specific compiler versions compatible with your CUDA toolkit (e.g., gcc-12/g++-12 for CUDA 12.x)
# sudo apt install gcc-12 g++-12
# export CC=gcc-12 CXX=g++-12 # Set compiler environment variables BEFORE building torchmd-net
```

## 🚀 Step 2: Install NVIDIA Driver and CUDA Toolkit

- **Install NVIDIA Driver:** Ensure you have a compatible NVIDIA driver installed for your GPU and CUDA version. Check NVIDIA's documentation for driver/CUDA compatibility. Installation methods vary (distribution packages, NVIDIA runfile).

- **Install CUDA Toolkit:** Download and install the appropriate CUDA Toolkit version (e.g., 12.4) from the NVIDIA developer website. Follow NVIDIA's installation instructions carefully. Example using runfile (adjust version numbers):

```bash
# wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
# chmod +x cuda_12.4.1_550.54.15_linux.run
# sudo sh cuda_12.4.1_550.54.15_linux.run --toolkit --silent --override
```
(Use flags like `--toolkit` to install only the toolkit if the driver is already installed. Refer to `sh cuda*.run --help`).

## 🔁 Step 3: Set CUDA Environment Variables

Set these variables in your shell environment. Add them to your `~/.bashrc` or `~/.zshrc` for persistence. Adjust the path if your CUDA installation differs.

```bash
export CUDA_HOME=/usr/local/cuda-12.4 # Adjust version if needed
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Optional: Specify CUDA compiler if needed by torchmd-net build
# export CUDACXX=$CUDA_HOME/bin/nvcc
```

Verify CUDA installation:

```bash
nvcc --version
nvidia-smi
```

## 🐍 Step 4: Create a Clean Conda/Mamba Environment

Using Mamba is recommended for faster dependency resolution.

```bash
# Using Mamba
mamba create -n poreanalysis-pocket python=3.9 -c conda-forge -y
mamba activate poreanalysis-pocket

# Or using Conda
# conda create -n poreanalysis-pocket python=3.9 -c conda-forge -y
# conda activate poreanalysis-pocket
```

## ⚙️ Step 5: Install PyTorch with Correct CUDA Support

Install PyTorch matching your installed CUDA Toolkit version. Check the official PyTorch website (https://pytorch.org/get-started/locally/) for the correct command.

Example for PyTorch 2.x and CUDA 12.x (verify command on PyTorch website):

```bash
# Using Mamba
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge --strict-channel-priority

# Or using Pip (ensure pip is up-to-date: python -m pip install --upgrade pip)
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
(Replace `cu121` with your specific CUDA version identifier if needed, e.g., `cu118`)

Verify PyTorch CUDA support:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

## 📦 Step 6: Install PyTorch Geometric and TorchMD-Net Dependencies

Install PyG first, following instructions on its documentation (https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) corresponding to your PyTorch and CUDA versions.   

Example using pip:

```bash
# Find the correct command for your PyTorch/CUDA versions on PyG website!
# Example for PyTorch 2.x, CUDA 12.1:
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html # Adjust torch version string!
```

Then, install TorchMD-Net dependencies (using Mamba/Conda is often easier):

```bash
# Activate your environment (e.g., poreanalysis-pocket)
# Navigate to the cloned PoreAnalysisSuite directory

# Using Mamba/Conda (preferred) - Installs most dependencies
mamba env update --name poreanalysis-pocket --file environment-pocket.yml # Assumes we create an environment-pocket.yml file OR add to base environment.yml
# OR install key dependencies manually:
# mamba install pytorch-lightning=1.8 hydra-core -c conda-forge

# Using Pip (install dependencies listed in torchmd-net setup/environment file)
# pip install pytorch-lightning==1.8 hydra-core # Example, check torchmd-net requirements
```
(Note: You might need to create an `environment-pocket.yml` or similar to list these dependencies for conda/mamba)

## 🛠️ Step 7: Install TorchMD-Net from Source

Clone the torchmd-net repository and install it from source to build the C++/CUDA extensions.

```bash
git clone https://github.com/torchmd/torchmd-net.git
cd torchmd-net

# IMPORTANT: Set compiler environment variables if needed (see Step 1)
# export CC=gcc-12
# export CXX=g++-12

# Install torchmd-net (builds extensions)
pip install -e .

cd .. # Go back to PoreAnalysisSuite directory
```

## ✅ Step 8: Install PoreAnalysisSuite with Pocket Extras

Now, install the Pore Analysis Suite with the pocket extras enabled:

```bash
# Ensure you are in the PoreAnalysisSuite root directory
# Ensure your conda environment (e.g., poreanalysis-pocket) is activated
pip install -e .[pocket]
```

## ✅ Step 9: Test Installation

Verify that the core torchmd-net module can be imported:

```bash
python -c "from torchmdnet.models.model import load_model; print('TorchMD-Net seems importable.')"
```

Verify the pocket_analysis module within the suite can be imported (this won't run analysis, just checks imports):

```bash
python -c "from pore_analysis.modules import pocket_analysis; print('PoreAnalysisSuite pocket module seems importable.')"
```

If these commands run without import errors, the advanced setup is likely complete. You can now run analyses using the `--pocket` flag.