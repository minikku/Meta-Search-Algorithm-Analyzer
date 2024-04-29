import os
from multiprocessing import Pool

import cma
import numpy as np
import warnings
import ioh
import nevergrad as ng
from .utilities import show_current_date_time

from ..algorithms.DCS import DCS
from nevergrad.optimization.optimizerlib import CMA, PSO, DE

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def run_parallel_function(runFunction, arguments, minPoolSize: int):
    arguments = list(arguments)
    p = Pool(min(minPoolSize, len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()

    print(show_current_date_time() + ' ' + 'COMPLETED 1/4: Raw results were obtained')

    return results


class AlgorithmEvaluator:
    def __init__(self, optimizer, bfac, _problem):
        self.alg = optimizer
        self.bfac = bfac
        self.problem_type = _problem
        self.custom_algorithm = ['DCS']  # Add your custom algorithms

    def __call__(self, func, seed):
        np.random.seed(int(seed))
        warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)

        lb = func.bounds.lb[0]
        ub = func.bounds.ub[0]
        max_budget = None

        parametrization = None
        if self.problem_type == 'BBOB':

            parametrization = ng.p.Array(shape=(func.meta_data.n_variables,)).set_bounds(lb, ub)
            max_budget = self.bfac * func.meta_data.n_variables

            # if self.alg not in self.custom_algorithm:
            #     max_budget = max_budget * 100

        elif self.problem_type == 'PBO':

            # parametrization = ng.p.Array(shape=(func.meta_data.n_variables,)).set_bounds(lb, ub)  # Continuous
            parametrization = ng.p.TransitionChoice(range(lb, ub + 1),
                                                    repetitions=func.meta_data.n_variables)  # Discrete
            # parametrization = ng.p.Choice(list(range(lb, ub + 1)), repetitions=func.meta_data.n_variables)  # Discrete
            max_budget = self.bfac

        if self.alg in self.custom_algorithm:
            optimizer = eval(f"{self.alg}")(parametrization=parametrization, problem_dim=func.meta_data.n_variables,
                                            problem_bounds=list([lb, ub]),
                                            optimization_type=func.meta_data.optimization_type.name,
                                            budget=int(max_budget))
        else:
            optimizer = eval(f"{self.alg}")(parametrization=parametrization,
                                            budget=int(max_budget))
        optimizer.minimize(func)
        # optimizer.provide_recommendation()


# Function to run the optimizer
def run_optimizer(temp):
    algname, fid, iid, dim, bfac, force_replace_flag, rep_num, _problem = temp
    # print(algname, 'F', fid, ' I', iid, 'D', dim)

    algorithm = AlgorithmEvaluator(algname, bfac, _problem)
    max_budget = bfac * dim

    func = None
    if _problem == 'BBOB':
        func = ioh.get_problem(fid=fid, dimension=dim, instance=iid, problem_class=ioh.ProblemClass.BBOB)
    elif _problem == 'PBO':
        func = ioh.get_problem(fid=fid, dimension=dim, instance=iid, problem_class=ioh.ProblemClass.PBO)

    fname = func.meta_data.name
    result_exist_flag = os.path.exists(
        './Results/' + _problem + '/' + algname + '_' + str(max_budget) + '_F' + str(fid) + '_I' + str(
            iid) + '_' + str(
            dim) + 'D/IOHprofiler_f' + str(fid) + '_' + fname + '.json')

    # Skip or not?
    if force_replace_flag or not result_exist_flag:
        logger = ioh.logger.Analyzer(root=f"./Results/{_problem}/",
                                     folder_name=f"{algname}_{max_budget}_F{fid}_I{iid}_{dim}D",
                                     algorithm_name=f"{algname}_{max_budget}",
                                     store_positions=True, triggers=[ioh.logger.trigger.ALWAYS])

        func.attach_logger(logger)
        for rep in range(rep_num):
            algorithm(func, rep)
            func.reset()
        logger.close()

    # else:
    #     print('SKIP: ' + './Results/' + algname + '_' + str(max_budget) + '_F' + str(fid) + '_I' + str(iid) + '_' + str(
    #         dim) + 'D/IOHprofiler_f' + str(fid) + '_' + fname + '.json')
