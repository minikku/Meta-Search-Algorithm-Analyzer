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

    def __init__(self, parametrization, budget=None, num_workers=1,
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

        # Extract problem's bounds
        self.lower_bound = self.parametrization.sample().bounds[0]
        self.upper_bound = self.parametrization.sample().bounds[1]
        self.lower_bound_vec = np.ones(self.dimension) * self.lower_bound
        self.upper_bound_vec = np.ones(self.dimension) * self.upper_bound

        for i in range(self.pop_size):
            data = self.parametrization.sample()
            # print(data.value)
            self.pos.append(data.value[:].tolist())
            self.offspring_pos.append(data.value[:].tolist())

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
            # Sort population by fitness values (best to worst: ascending)
            # print([self.pos, self.fitness])
            self.pos, self.fitness = PopSort(self.pos, self.fitness, 'ascending')
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
        jrand = np.floor(self.dimension * np.random.rand()).astype(int)
        self.offspring_pos[i, :] = self.pos[i, :]

        if i == self.pop_size - 1 and np.random.rand() < 0.5:
            # Low-performing
            self.offspring_pos[i, :] = self.lower_bound_vec + np.random.rand() * (self.upper_bound_vec - self.lower_bound_vec)
        elif i < self.ngS - 1:
            # High-performing
            r1 = i
            while r1 == i or r1 == self.bestInd:
                r1 = np.floor(self.pop_size * np.random.rand()).astype(int)

            for d in range(self.dimension):
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

            for d in range(self.dimension):
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
        if value <= self.fitness[i]:

            # print('old pos[',i,']: ', self.pos[i, :], 'value: ', self.fitness[i])
            self.fitness[i] = value
            self.pos[i, :] = x
            # print('new pos[',i,']: ', self.pos[i, :], 'value: ', self.fitness[i])

            if value < self.best_fitness:
                self.best_fitness = value
                self.bestInd = i

                # print('best pos[', self.bestInd, ']: ', self.pos[self.bestInd, :], 'value: ', self.best_fitness)

        if self.current_i < self.pop_size-1:
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
            nevergrad_array = ng.p.Array(init=self.offspring_pos[current_i, :]).set_bounds(lower=self.lower_bound, upper=self.upper_bound)
            # print('curr ind', current_i, ': ', nevergrad_array)

        return nevergrad_array

    # def _tell(self, x, value):
    #     print('_tell')
    #     # Update the fitness of the evaluated candidate
    #     print('x', x)
    #     for i, pos_i in enumerate(self.pos):
    #         if value[i] <= self.fitness[i]:
    #             self.fitness[i] = value[i]
    #             self.pos[i, :] = self.offspring_pos[i, :]  #x[i, :].value
    #
    #             if value[i] < self.best_fitness:
    #                 self.best_fitness = value[i]
    #                 self.bestInd = i
    #         break
    #
    #     for i in range(self.pop_size):
    #         self.current_i = i
    #         self._internal_update()
    #
    #     return self.fitness

    # def __call__(self, func):
    #     self.best_x = None
    #     self.best_cost = np.Inf
    #     # def DCS(search_agent_no, max_nfe, lb, ub, dim, fobj):
    #     # Parameters
    #     NP = self.popsize  # 30
    #     D = func.meta_data.n_variables
    #     L = func.bounds.lb
    #     U = func.bounds.ub
    #     next_pos = np.zeros((NP, D))
    #     new_pos = np.zeros((NP, D))
    #     max_nfe = self.budget
    #     max_itr = round(max_nfe / NP)
    #     qCR = np.zeros(NP)
    #     new_fitness = np.zeros(NP)
    #
    #     # Evaluation settings
    #     # eval_setting = {'case': fobj, 'lb': L, 'ub': U, 'dim': dim}
    #     cul_solution = []
    #     cul_nfe = []
    #     convergence_curve = []
    #
    #     # Golden ratio
    #     golden_ratio = 2 / (1 + sqrt(5))
    #
    #     # High-performing individuals
    #     ngS = round(NP * self.p)  # max(6, round(NP * (golden_ratio / 3)))
    #
    #     # Initialize the population
    #     pos = L + np.random.rand(NP, D) * (U - L)  # Real-valued population
    #     # pos_bin = np.round(pos)  # Binary population
    #
    #     # Initialize fitness values
    #     fitness = np.zeros(NP)
    #     for i in range(NP):
    #         fitness[i] = func(pos[i, :])
    #         # fitness[i] = func(pos_bin[i, :].astype(int))
    #
    #     # Generation
    #     nfe = 0
    #
    #     # Ranking-based self-improvement
    #     # rank_to_qCR = 0.25 + 0.55 * (np.arange(1, NP + 1) / NP) ** 0.5
    #     rank_to_qCR = self.phi[0] + self.phi[1] * (np.arange(1, NP + 1) / NP) ** 0.5
    #
    #     while nfe < max_nfe:
    #
    #         # Sort population by fitness values (best to worst: ascending)
    #         # pos, fitness = PopSort(pos, fitness, 'descending')
    #         # pos_bin, fitness = PopSort(pos_bin, fitness, 'descending')
    #         pos, fitness = PopSort(pos, fitness, 'ascending')
    #
    #         # Best solution
    #         best_fitness = fitness[0]
    #         bestInd = 0
    #
    #         # Compute social impact factor
    #         # social_impact = 0.1 + (0.518 * (1 - (nfe / max_nfe) ** 0.5))
    #         social_impact = self.lamda[0] + (self.lamda[1] * (1 - (nfe / max_nfe) ** 0.5))
    #
    #         for i in range(NP):
    #
    #             learning_ability = np.random.rand()
    #
    #             # Compute differentiated knowledge-acquisition rate
    #             qCR[i] = (np.round(np.random.rand() * rank_to_qCR[i]) + (np.random.rand() <= rank_to_qCR[i])) / 2
    #             jrand = np.floor(D * np.random.rand()).astype(int)
    #             next_pos[i, :] = pos[i, :]
    #
    #             if i == NP - 1 and np.random.rand() < 0.5:
    #                 # Low-performing
    #                 next_pos[i, :] = L + np.random.rand() * (U - L)
    #             elif i < ngS - 1:
    #                 # High-performing
    #                 r1 = i
    #                 while r1 == i or r1 == bestInd:
    #                     r1 = np.floor(NP * np.random.rand()).astype(int)
    #
    #                 for d in range(D):
    #                     if np.random.rand() <= qCR[i] or d == jrand:
    #                         # next_pos[i, d] = pos[r1, d] + LnF(golden_ratio, 0.05, 1, 1)
    #                         next_pos[i, d] = pos[r1, d] + LnF(self.alpha, self.sigma, 1, 1)
    #             else:
    #                 # Average-performing
    #                 r1, r2, r3 = i, i, i
    #                 while r1 == i or r1 == bestInd:
    #                     r1 = np.floor(NP * np.random.rand()).astype(int)
    #                 while r2 == i or r2 == bestInd or r2 == r1:
    #                     r2 = ngS + np.floor((NP - ngS) * np.random.rand()).astype(int)
    #                 while r3 == i:
    #                     r3 = np.floor(NP * np.random.rand()).astype(int)
    #
    #                 # learning_ability = np.random.rand()
    #                 # x_rand = np.random.rand()
    #
    #                 for d in range(D):
    #                     if np.random.rand() <= qCR[i] or d == jrand:
    #                         next_pos[i, d] = self.w * pos[bestInd, d] + (
    #                                 (pos[r1, d] - pos[i, d]) * learning_ability) + (
    #                                                  (pos[r2, d] - pos[i, d]) * social_impact)
    #
    #             # Boundary
    #             # next_pos[i, :] = boundConstraint(next_pos[i, :], pos[i, :], np.array([L, U]))
    #             next_pos[i, :] = np.where(next_pos[i, :] < L, (pos[i, :] + L) / 2, next_pos[i, :])
    #             next_pos[i, :] = np.where(next_pos[i, :] > U, (pos[i, :] + U) / 2, next_pos[i, :])
    #
    #             pos_tmp = next_pos[i, :][:]  # Trial vector
    #             # pos_tmp = np.round(next_pos[i, :])
    #
    #             # NFE increment with condition
    #             # we enforce that offspring created by mutation are different from their parent
    #             # and resample without further evaluation if needed.
    #             # if pos_tmp.all() != pos[i, :].all():
    #             # if not np.all(pos_tmp == pos[i, :]):
    #             if True:
    #                 # new_fitness[i] = func(pos_tmp.astype(int))
    #                 new_fitness[i] = func(pos_tmp)
    #
    #                 nfe += 1
    #
    #                 if new_fitness[i] <= fitness[i]:  # Maximization
    #                     pos[i, :] = next_pos[i, :][:]  # Update real-valued population
    #                     fitness[i] = new_fitness[i]
    #                     # pos_bin[i, :] = pos_tmp[:]  # Update binary population
    #
    #                     if new_fitness[i] < best_fitness:  # Maximization
    #                         best_fitness = new_fitness[i]
    #                         bestInd = i
    #                         self.tracked_bestInd = bestInd
    #                         self.tracked_learning_ability = learning_ability
    #                         self.tracked_social_impact = social_impact
    #
    #         self.best_x = pos[bestInd, :]
    #         self.best_cost = best_fitness
    #
    #         # print('NFE: ', nfe, '(running)')
    #         # print('Best cost: ', self.best_cost)
    #
    #         convergence_curve.append(best_fitness)
    #         cul_solution.append(self.best_x.tolist())
    #         cul_nfe.append(nfe)
    #
    #     return self.best_cost, self.best_x, convergence_curve, cul_solution, cul_nfe

    @property
    def a_property(self):
        return np.random.randint(100)

    def reset(self):
        self.algorithm_id = np.random.randint(100)


#################################################################################
# Register the optimizer (optional, for convenience)
registry.register(DCS)
