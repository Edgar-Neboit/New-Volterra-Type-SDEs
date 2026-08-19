# Numerical Simulation of Volterra-Type SDEs

Repository containing the numerical implementation of the discrete scheme presented in the paper **Efficient simulation of a new class of Volterra-type SDEs** (see arXiv preprint: [arXiv:2306.02708](https://arxiv.org/abs/2306.02708)) by Ofelia Bonesini, Giorgia Callegaro, Martino Grasselli, and Gilles Pagès. 

The implementation relies heavily on the Python library Numba for Just-In-Time (JIT) compilation, ensuring high performance computation without requiring GPU hardware. The dependencies are detailed in the associated `requirements.txt` file. 

## Getting Started

### 1. Clone the Repository
Open your terminal and run the following commands to clone the project to your local machine:

```bash
git clone [https://github.com/your-username/New-Volterra-Type-SDEs.git](https://github.com/your-username/New-Volterra-Type-SDEs.git)
cd your-repo-name
```

It is highly recommended to use a virtual environment (Conda or venv) to manage dependencies. 
```bash
# Using venv
# Create the environment
python -m venv PyEnv

# Activate the environment
# On Windows:
.\PyEnv\Scripts\activate
# On macOS/Linux:
source PyEnv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

With conda : 
```bash
# Create the environment (Python 3.9+ recommended)
conda create --name MyEnv python=3.9

# Activate the environment
conda activate MyEnv

# Install dependencies
pip install -r requirements.txt

