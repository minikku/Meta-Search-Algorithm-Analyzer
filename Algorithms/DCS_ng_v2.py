# %pip install ioh
# %pip install scipy

import numpy as np
from nevergrad.common import typing as tp
from scipy.stats import laplace
from scipy.special import gamma
from math import pi, sin, cos, tan, sqrt, log

from ioh import problem, get_problem, ProblemClass
from ioh import logger

from ioh import Experiment

import random
from multiprocessing import Pool, freeze_support
import timeit
from datetime import datetime
import time

import nevergrad as ng
from nevergrad.optimization.optimizerlib import base, registry


# import warnings
# warnings.filterwarnings('ignore')

# help(Experiment)

# Random generator seed
# random.seed(42)  # Set the seed for the random number generator
# np.random.seed(42)  # Set the seed for numpy's random number generator


# Additional functions for support
def PopSort(input_pop, input_fitness, sort_style):
    if sort_style == 'ascending':
        sorted_index = input_fitness.argsort()
        sorted_population = input_pop[sorted_index, :]
        sorted_fitness = input_fitness[sorted_index]
    elif sort_style == 'descending':
        n = input_fitness.shape[0]
        sorted_index = input_fitness.argsort()[::-1][:n]
        sorted_population = input_pop[sorted_index, :]
        sorted_fitness = input_fitness[sorted_index]
    return sorted_population, sorted_fitness


def LnF(alpha, scale, m, n):
    xhold = laprnd(m, n, 0, 1)
    SE = np.sign(np.random.rand(m, n) - 0.5) * xhold
    U = np.random.rand(m, n)
    xhold = (sin(0.5 * pi * alpha) * tan(0.5 * pi * (1 - alpha * U)) - cos(0.5 * pi * alpha)) ** (1 / alpha)
    xhold = scale * SE / xhold
    return xhold


def LnF2(alpha, scale, m, n):
    xhold = laprnd(m, n, 0, 1)
    SE = np.sign(np.random.rand(m, n) - 0.5) * xhold
    U = np.random.rand(m, n)
    xhold = (sin(0.5 * pi * alpha) * 1 / tan(0.5 * pi * (alpha * U)) - cos(0.5 * pi * alpha)) ** (1 / alpha)
    xhold = scale * SE / xhold
    return xhold


def LnF3(alpha, sigma, m, n):
    Z = laplacernd(m, n)
    Z = np.sign(np.random.rand(m, n) - 0.5) * Z
    U = np.random.rand(m, n)
    R = sin(0.5 * pi * alpha) * tan(0.5 * pi * (1 - alpha * U)) - cos(0.5 * pi * alpha)
    Y = sigma * Z * (R) ** (1 / alpha)
    return Y


def laplacernd(m, n):
    u1 = np.random.rand(m, n)
    u2 = np.random.rand(m, n)
    x = log(u1 / u2)
    return x


def laprnd(m, n, mu=0, sigma=1):
    u = np.random.rand(m, n) - 0.5
    b = sigma / sqrt(2)
    y = mu - b * np.sign(u) * log(1 - 2 * np.abs(u))
    return y


def LvF(beta, scale, n, d):
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (
            1 / beta)
    u = np.random.randn(n, d) * sigma
    v = np.random.randn(n, d)
    step = u / (np.abs(v) ** (1 / beta))
    o = scale * step
    return o


def LiC(beta, scale, row, col):
    xLp = np.random.rand(row, col) - 0.5
    xLp = - sqrt(beta) * np.sign(xLp) * log(1 - 2 * np.abs(xLp))
    xLk = tan(0.5 * pi * (1 - np.random.rand(row, col))) ** (1 / beta)
    out = scale * xLp / xLk
    return out


def LiCv2(beta, scale, row, col):
    xLp = np.random.rand(row, col) - 0.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (
            1 / beta)
    xLp = - sigma * 0.7071 * np.sign(xLp) * log(1 - 2 * np.abs(xLp))
    xLk = tan(0.5 * pi * (1 - np.random.rand(row, col))) ** (1 / beta)
    out = scale * xLp / xLk
    return out


def boundConstraint(vi, pop, lu):
    # NP, D = pop.shape
    NP = len(pop)
    D = pop.shape[0]
    xl = np.tile(lu[0, :], (NP, 1))
    xu = np.tile(lu[1, :], (NP, 1))
    vi = np.where(vi < xl, (pop + xl) / 2, vi)
    vi = np.where(vi > xu, (pop + xu) / 2, vi)
    return vi


def boundConstraint2(vi, pop, lu):
    NP, D = pop.shape
    xl = np.tile(lu[0, :], (NP, 1))
    xu = np.tile(lu[1, :], (NP, 1))
    vi = np.where(vi < xl, pop, vi)
    vi = np.where(vi > xu, pop, vi)
    return vi


def boundConstraint3(vi, mean_pos, lu):
    xl = np.tile(lu[0, :], (1, 1))
    xu = np.tile(lu[1, :], (1, 1))
    vi = np.where(vi < xl, mean_pos, vi)
    vi = np.where(vi > xu, mean_pos, vi)
    return vi


class DCS(base.Optimizer):

    def __init__(self, parametrization, problem_dim, problem_bounds, optimization_type, budget=None, num_workers=1,
                 inp_popsize=40, inp_p=.618, inp_alpha=.618,
                 inp_sigma=.05,
                 inp_w=1,
                 inp_lamda=None, inp_omega=None, inp_phi=None
                 ):
        super().__init__(parametrization, budget=budget, num_workers=num_workers)

        # print('_init')
        # Note that we should re-initialize all dynamic variables if we want to run the same algorithm multiple times
        if inp_phi is None:
            inp_phi = [0.25, 0.55]
        if inp_lamda is None:
            inp_lamda = [0.1, 0.518]
        if inp_omega is None:
            inp_omega = []

        self.budget = budget  # Max NFE
        self.p = inp_p
        self.alpha = inp_alpha
        self.sigma = inp_sigma
        self.w = inp_w
        self.lamda = inp_lamda
        self.phi = inp_phi
        self.pop_size = inp_popsize
        self.omega = inp_omega

        # A parameter static over the course of an optimization run of an algorithm
        self.algorithm_id = np.random.randint(100)

        # A dynamic parameter updated by the algorithm
        self.tracked_bestInd = None
        self.tracked_learning_ability = None
        self.tracked_social_impact = None

        self.best_x = None
        self.best_cost = np.Inf

        # Parameters
        self.max_nfe = self.budget
        self.nfe = 0

        # Golden ratio
        self.golden_ratio = 2 / (1 + sqrt(5))

        # High-performing individuals
        self.ngS = round(self.pop_size * self.p)  # max(6, round(NP * (golden_ratio / 3)))

        # Initialize the population
        self.pos = []
        self.offspring_pos = []
        # self.fitness = []

        self.dim = problem_dim

        # Extract problem's bounds
        self.lower_bound = problem_bounds[0]  # self.parametrization.sample().bounds[0]
        self.upper_bound = problem_bounds[1]  # self.parametrization.sample().bounds[1]
        self.lower_bound_vec = np.ones(self.dim) * self.lower_bound
        self.upper_bound_vec = np.ones(self.dim) * self.upper_bound

        self.optimization_type = optimization_type

        for i in range(self.pop_size):
            data = self.parametrization.sample()
            # print(data.value)
            # self.pos.append(data.value[:].tolist())
            self.pos.append(list(data.value[:]))
            # self.offspring_pos.append(data.value[:].tolist())
            self.offspring_pos.append(list(data.value[:]))

        # Convert list to numpy array
        self.pos = np.array(self.pos)
        self.offspring_pos = np.array(self.offspring_pos)

        self.discount_nfe_for_initial_phase = True

        # Initialize fitness values
        self.fitness = np.ones(self.pop_size) * np.Inf
        for i in range(self.pop_size):
            # print('itr->', i)
            self._ask(i)

        self.discount_nfe_for_initial_phase = False

        # Ranking-based self-improvement
        # rank_to_qCR = 0.25 + 0.55 * (np.arange(1, NP + 1) / NP) ** 0.5
        self.rank_to_qCR = self.phi[0] + self.phi[1] * (np.arange(1, self.pop_size + 1) / self.pop_size) ** 0.5

        self.bestInd = 0
        self.best_fitness = np.Inf

        self.social_impact = None
        self.learning_ability = None
        self.qCR = np.array(np.zeros(self.pop_size))

        self.current_i = 0

        # print('pos_init', self.pos)

    def _internal_update(self):
        # print('_internal_update for ', self.current_i)

        # Sort once per generation
        if self.current_i == 0:
            # Sort population by fitness values (best to worst)
            # print([self.pos, self.fitness])
            if self.optimization_type == 'MIN':
                self.pos, self.fitness = PopSort(self.pos, self.fitness, 'ascending')
            elif self.optimization_type == 'MAX':
                self.pos, self.fitness = PopSort(self.pos, self.fitness, 'descending')
            # print([self.pos, self.fitness])

        # Best solution
        best_fitness = self.fitness[0]
        self.bestInd = 0

        # Compute social impact factor
        # social_impact = 0.1 + (0.518 * (1 - (nfe / max_nfe) ** 0.5))
        self.social_impact = self.lamda[0] + (self.lamda[1] * (1 - (self.nfe / self.max_nfe) ** 0.5))

        i = self.current_i

        # for i in range(self.pop_size):

        self.learning_ability = np.random.rand()

        # Compute differentiated knowledge-acquisition rate
        self.qCR[i] = (np.round(np.random.rand() * self.rank_to_qCR[i]) + (
                np.random.rand() <= self.rank_to_qCR[i])) / 2
        jrand = np.floor(self.dim * np.random.rand()).astype(int)
        self.offspring_pos[i, :] = self.pos[i, :]

        if i == self.pop_size - 1 and np.random.rand() < 0.5:
            # Low-performing
            self.offspring_pos[i, :] = self.lower_bound_vec + np.random.rand() * (
                        self.upper_bound_vec - self.lower_bound_vec)
        elif i < self.ngS - 1:
            # High-performing
            r1 = i
            while r1 == i or r1 == self.bestInd:
                r1 = np.floor(self.pop_size * np.random.rand()).astype(int)

            for d in range(self.dim):
                if np.random.rand() <= self.qCR[i] or d == jrand:
                    # next_pos[i, d] = pos[r1, d] + LnF(golden_ratio, 0.05, 1, 1)
                    self.offspring_pos[i, d] = self.pos[r1, d] + LnF(self.alpha, self.sigma, 1, 1)
        else:
            # Average-performing
            r1, r2, r3 = i, i, i
            while r1 == i or r1 == self.bestInd:
                r1 = np.floor(self.pop_size * np.random.rand()).astype(int)
            while r2 == i or r2 == self.bestInd or r2 == r1:
                r2 = self.ngS + np.floor((self.pop_size - self.ngS) * np.random.rand()).astype(int)
            while r3 == i:
                r3 = np.floor(self.pop_size * np.random.rand()).astype(int)

            # learning_ability = np.random.rand()
            # x_rand = np.random.rand()

            for d in range(self.dim):
                if np.random.rand() <= self.qCR[i] or d == jrand:
                    self.offspring_pos[i, d] = self.w * self.pos[self.bestInd, d] + (
                            (self.pos[r1, d] - self.pos[i, d]) * self.learning_ability) + (
                                                       (self.pos[r2, d] - self.pos[i, d]) * self.social_impact)

        # Boundary
        # next_pos[i, :] = boundConstraint(next_pos[i, :], pos[i, :], np.array([L, U]))
        self.offspring_pos[i, :] = np.where(self.offspring_pos[i, :] < self.lower_bound_vec,
                                            (self.pos[i, :] + self.lower_bound_vec) / 2,
                                            self.offspring_pos[i, :])
        self.offspring_pos[i, :] = np.where(self.offspring_pos[i, :] > self.upper_bound_vec,
                                            (self.pos[i, :] + self.upper_bound_vec) / 2,
                                            self.offspring_pos[i, :])

        nevergrad_array = ng.p.Array(init=self.offspring_pos[i, :]).set_bounds(lower=self.lower_bound,
                                                                               upper=self.upper_bound)
        # print(nevergrad_array)
        return nevergrad_array.value

    def _internal_ask(self):
        # print('_internal_ask for ', self.current_i)
        return self._internal_update()

    def _internal_tell(self, x, value):
        # print('_internal_tell for ', self.current_i)
        # Update the fitness of the evaluated candidate
        # print('x', x, 'value: ', value)
        i = self.current_i

        if self.optimization_type == 'MIN':
            if value <= self.fitness[i]:

                # print('old pos[',i,']: ', self.pos[i, :], 'value: ', self.fitness[i])
                self.fitness[i] = value
                self.pos[i, :] = x
                # print('new pos[',i,']: ', self.pos[i, :], 'value: ', self.fitness[i])

                if value < self.best_fitness:
                    self.best_fitness = value
                    self.bestInd = i

                    # print('best pos[', self.bestInd, ']: ', self.pos[self.bestInd, :], 'value: ', self.best_fitness)

        elif self.optimization_type == 'MAX':
            if value >= self.fitness[i]:

                # print('old pos[',i,']: ', self.pos[i, :], 'value: ', self.fitness[i])
                self.fitness[i] = value
                self.pos[i, :] = x
                # print('new pos[',i,']: ', self.pos[i, :], 'value: ', self.fitness[i])

                if value > self.best_fitness:
                    self.best_fitness = value
                    self.bestInd = i

                    # print('best pos[', self.bestInd, ']: ', self.pos[self.bestInd, :], 'value: ', self.best_fitness)

        if self.current_i < self.pop_size - 1:
            self.current_i = self.current_i + 1
            # print('next ind: ', self.current_i)
        else:
            self.current_i = 0

        # NFE increment
        if self.nfe < self.max_nfe and not self.discount_nfe_for_initial_phase:
            self.nfe = self.nfe + 1

    def _ask(self, current_i):
        # print('_ask')
        # Here, we would normally decide which candidate to evaluate next.
        # self.nfe = self.nfe + 1
        if current_i is None:
            # nevergrad_array = ng.p.Array(init=self.offspring_pos[self.bestInd, :]).set_bounds(lower=self.lower_bound, upper=self.upper_bound)
            nevergrad_array = ng.p.Array(init=self.pos[self.bestInd, :]).set_bounds(lower=self.lower_bound,
                                                                                    upper=self.upper_bound)
            # print('best ind', self.bestInd, ': ', nevergrad_array)
        else:
            # self.nfe = self.nfe + 1
            nevergrad_array = ng.p.Array(init=self.offspring_pos[current_i, :]).set_bounds(lower=self.lower_bound,
                                                                                           upper=self.upper_bound)
            # print('curr ind', current_i, ': ', nevergrad_array)

        return nevergrad_array

    @property
    def a_property(self):
        return np.random.randint(100)

    def reset(self):
        self.algorithm_id = np.random.randint(100)


#################################################################################
# Register the optimizer (optional, for convenience)
registry.register(DCS)
