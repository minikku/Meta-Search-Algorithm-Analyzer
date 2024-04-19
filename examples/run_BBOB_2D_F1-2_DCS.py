from msaa.core.pipeline import run_pipeline
from multiprocessing import freeze_support
from msaa.algorithms.DCS import DCS

if __name__ == '__main__':
    freeze_support()

    # ALGORITHMS
    ng_algs = ['DCS']

    # FUNCTIONS
    fids = [1, 2]

    # INSTANCES
    iids = [1]

    # DIMENSIONS
    dims = [2]

    # BUDGETS FOR EACH DIMENSION
    bfacs = [10000]

    # PARALLEL WORKERS
    pool_size = 6

    # RE-RUN THE EXPERIMENT
    force_replace_old_results = False

    # PER ALGORITHM / FUNCTION
    repetition = 1

    # PROBLEM TYPE: BBOB, PBO
    problem_type = ['BBOB']

    # Execute all procedures
    run_pipeline(ng_algs, fids, iids, dims, bfacs, force_replace_old_results, repetition, problem_type, pool_size)
