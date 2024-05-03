from msaa.core.pipeline import run_pipeline
from multiprocessing import freeze_support

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

if __name__ == '__main__':
    freeze_support()

    # ALGORITHMS
    ng_algs = ['DCS', 'PSO', 'DE', 'ES']

    # FUNCTIONS
    fids = [1, 2]

    # INSTANCES
    iids = [1, 2, 3, 4, 5]

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

    # ML MODEL OPTIONS
    models = [
        LinearRegression(),
        Ridge(),
        Lasso(),
        ElasticNet(),
        KNeighborsRegressor(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        SVR()
    ]

    # Execute all procedures
    run_pipeline(ng_algs, fids, iids, dims, bfacs, force_replace_old_results, repetition, problem_type, pool_size, models)
