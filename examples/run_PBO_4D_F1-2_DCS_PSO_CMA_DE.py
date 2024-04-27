from msaa.core.pipeline import run_pipeline
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    # ALGORITHMS
    ng_algs = ['DCS', 'PSO', 'CMA', 'DE']

    # FUNCTIONS
    fids = [1, 2]

    # INSTANCES
    iids = [1]

    # DIMENSIONS
    dims = [4]

    # BUDGETS FOR EACH DIMENSION
    bfacs = [1000]

    # PARALLEL WORKERS
    pool_size = 26

    # RE-RUN THE EXPERIMENT
    force_replace_old_results = False

    # PER ALGORITHM / FUNCTION
    repetition = 1

    # PROBLEM TYPE: BBOB, PBO
    problem_type = ['PBO']

    # Execute all procedures
    run_pipeline(ng_algs, fids, iids, dims, bfacs, force_replace_old_results, repetition, problem_type, pool_size)
