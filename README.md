# Search Algorithm Analyzer

This project provides a Python-based analysis tool tailored for optimizing algorithm performance across both continuous and discrete problem domains, utilizing the Nevergrad optimization framework and exploratory landscape analysis (ELA). It specifically targets the Black-Box Optimization Benchmarking (BBOB) and Pseudo-Boolean Optimization (PBO) suites, which are crucial for evaluating the proficiency of optimization algorithms. BBOB assesses algorithms on continuous optimization problems through a diverse set of functions, highlighting challenges in robustness, efficiency, and scalability across varied problem landscapes. Conversely, PBO focuses on discrete optimization, examining how algorithms navigate complex binary vectors, thereby offering a comprehensive view of algorithmic behavior and performance across a spectrum of combinatorial structures and difficulties. The core mission of this project is to enhance the development and refinement of optimization algorithms, ensuring adeptness in tackling an extensive array of optimization challenges within these well-established benchmarking frameworks.

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
This will initiate the optimization process, analyzing algorithms' performance across specified problem domains.

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
- **Thomas Bäck** - *Leiden Institute of Advanced Computer Science, Leiden University, Netherlands*
- **Jan van Rijn** - *Leiden Institute of Advanced Computer Science, Leiden University, Netherlands*
- **Diederick Vermetten** - *Leiden Institute of Advanced Computer Science, Leiden University, Netherlands*
- **Khamron Sunat** - *College of Computing, Khon Kaen University, Thailand*
- **Poomin Duankhan** - *College of Computing, Khon Kaen University, Thailand*

## License
Copyright (c) 2024 Poomin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
