# Meta-Search Algorithm Analyzer (MSAA)

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

This project provides a Python-based analysis tool tailored for optimizing algorithm performance across both continuous and discrete problem domains, utilizing the Nevergrad optimization framework and exploratory landscape analysis (ELA). It specifically targets the Black-Box Optimization Benchmarking (BBOB) and Pseudo-Boolean Optimization (PBO) suites, which are crucial for evaluating the proficiency of optimization algorithms. BBOB assesses algorithms on continuous optimization problems through a diverse set of functions, highlighting challenges in robustness, efficiency, and scalability across varied problem landscapes. Conversely, PBO focuses on discrete optimization, examining how algorithms navigate complex binary vectors, thereby offering a comprehensive view of algorithmic behavior and performance across a spectrum of combinatorial structures and difficulties. The core mission of this project is to enhance the development and refinement of optimization algorithms, ensuring adeptness in tackling an extensive array of optimization challenges within these well-established benchmarking frameworks.

## Key Features

- **Nevergrad Integration:** Seamlessly works with Nevergrad, allowing access to a vast array of optimization algorithms suited for various tasks.
- **BBOB and PBO Support:** Includes dedicated support for the renowned BBOB and PBO benchmark suites, enabling thorough evaluation of algorithmic performance across a wide range of problem types.
- **Customizable Algorithms:** Users can easily configure the suite to include both Nevergrad's built-in algorithms and any custom optimizers, placing them in the 'Algorithms' directory for immediate use.
- **Exploratory Landscape Analysis:** Utilizes ELA to delve into the intricacies of problem landscapes, aiding in the strategic selection and development of optimization algorithms.
- **Flexible Experimentation:** Offers an array of customization options, from algorithm selection to problem dimensionality, ensuring that researchers can tailor their experiments to meet specific research objectives.
- **User-Centric Documentation:** Comes with detailed documentation, including setup instructions and usage guidelines, ensuring a smooth start and ongoing ease of use for both new and experienced users.
- **Analytical Tools for Evaluation:** Beyond optimization tasks, the framework includes tools for data preprocessing, feature computation, and performance evaluation, providing a full spectrum of research capabilities.
- **Visualization and Analysis:** Recommends methods for result analysis and includes visualization tools, aiding in the clear presentation and interpretation of experimental outcomes.
- **Community Contributions:** Encourages users to contribute by adding new algorithms, and problem definitions, or enhancing existing features, fostering a collaborative and ever-evolving project environment.
    
## Installation

### Requirements

- Python 3.9 or higher
- pip (Python package installer)
  
[![Python](https://skillicons.dev/icons?i=python)](https://www.python.org/)

### Setting Up a Virtual Environment

To avoid conflicts with other Python projects or system-wide packages, it's recommended to use a virtual environment. Create and activate one as follows:

#### For Unix/Linux/MacOS
```bash
python3 -m venv venv
source venv/bin/activate
```

#### For Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```
#### Installing Dependencies

With the virtual environment activated, install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

After installing the required packages, you can run the code by executing:
```bash
python Run.py
```
This command will initiate the optimization process, analyzing algorithms' performance across specified problem domains. You can customize the settings below according to your needs.
```bash
# ---------------------- Main execution block ---------------------------

# ALGORITHMS - List of algorithms to test. Can be customized as needed.
ng_algs = ['DCS']  # Compatible with Nevergrad's built-in and custom optimizers (place in the 'Algorithms' directory).

# FUNCTIONS - List of function IDs to evaluate. Adjust according to the benchmark suite.
# fids for BBOB might include [1, 2, 3, ..., 24]. Here, using PBO as an example.
fids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

# INSTANCES - Which instances of each function to run. Customize as per requirement.
iids = [1, 2, 3, 4, 5]

# DIMENSIONS - Dimensions of the problem to be tested. Varies by problem type.
dims = [16]  # Example for PBO. For BBOB, might include [2, 5, 10, 20, 40].

# BUDGETS FOR EACH DIMENSION - Number of function evaluations. Adjust based on computational resources.
bfacs = [1000]  # Example for PBO. For BBOB, might be higher, e.g., [10000].

# PARALLEL WORKERS - Number of parallel processes to use. Depends on the machine's capability.
pool_size = 26

# RE-RUN THE EXPERIMENT - Set to True to force rerun experiments even if results exist.
force_replace_old_results = False

# PER ALGORITHM / FUNCTION - Number of repetitions for each algorithm/function combination.
repetition = 5

# PROBLEM TYPE - Indicates the benchmark suite to use. ['BBOB', 'PBO'] are options.
problem_type = ['PBO']  # Can be set to ['BBOB', 'PBO'] to run both suites.
```

## Main Components
Function **`run_parallel_function`** is responsible for executing tasks in parallel.
```bash
def run_parallel_function(runFunction, arguments, minPoolSize: int):
    """
    Executes tasks in parallel based on the provided function and arguments.
    
    :param runFunction: The function to be executed in parallel.
    :param arguments: A list of arguments for the function calls.
    :param minPoolSize: The minimum size of the process pool.
    """
```

The **`AlgorithmEvaluator`** class wraps the algorithm optimization process, providing a callable interface for evaluating algorithm performance.
```bash
class AlgorithmEvaluator:
    def __init__(self, optimizer, bfac, _problem):
        """
        Initializes the evaluator with the optimizer, budget factor, and problem type.
        
        :param optimizer: The optimization algorithm.
        :param bfac: The budget factor, determining the number of evaluations.
        :param _problem: The problem domain (e.g., 'BBOB', 'PBO').
        """
```

Function **`compute_ela`** computes ELA features for a given set of input data, facilitating landscape analysis.
```bash
def compute_ela(X, y, min_y, max_y, lower_bound, upper_bound):
    """
    Computes ELA features for the given data.
    
    :param X: Input features.
    :param y: Target values.
    :param min_y: Minimum target value.
    :param max_y: Maximum target value.
    :param lower_bound: Lower bound of the input space.
    :param upper_bound: Upper bound of the input space.
    """
```

## Contributors ✨

This project exists thanks to all the people who contribute.
- **Poomin Duankhan** - *College of Computing, Khon Kaen University, Thailand*
- **Diederick Vermetten** - *Leiden Institute of Advanced Computer Science, Leiden University, Netherlands*
- **Thomas Bäck** - *Leiden Institute of Advanced Computer Science, Leiden University, Netherlands*
- **Jan van Rijn** - *Leiden Institute of Advanced Computer Science, Leiden University, Netherlands*
- **Khamron Sunat** - *College of Computing, Khon Kaen University, Thailand*
