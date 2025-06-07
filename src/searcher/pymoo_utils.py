from typing import Callable, Optional, Sequence, Union

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling


class WrappedPymooProblem(Problem):
    def __init__(
        self,
        n_var: int,
        score_fn: Callable[[np.ndarray], np.ndarray],
        MAXIMIZE: bool = True,
        xl: Optional[Union[Sequence[float], np.ndarray]] = None,
        xu: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> None:
        self.score_fn = score_fn
        self.MAXIMIZE = MAXIMIZE
        super().__init__(
            n_var=n_var,
            n_obj=1,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        assert x.shape[1] == self.n_var
        score = self.score_fn(x)
        assert score.shape[0] == x.shape[0]
        assert score.shape[1] == self.n_obj
        if self.MAXIMIZE:
            score = (-1) * score
        out["F"] = score


class CategoricalSampling(Sampling):
    def __init__(self, initial_data: Optional[np.ndarray] = None):
        super().__init__()
        self.initial_data = initial_data

    def _do(self, problem, n_samples, **kwargs):
        if self.initial_data is not None and len(self.initial_data) >= n_samples:
            return self.initial_data[:n_samples]

        return np.random.randint(
            problem.xl, problem.xu + 1, size=(n_samples, problem.n_var)
        )


class UniformCrossover(Crossover):
    def __init__(self, prob_c=0.9):
        super().__init__(2, 2)  # 2 parents, 2 offspring
        self.prob_c = prob_c

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        # Decide which designs to crossover
        do_crossover = np.random.random(n_matings) < self.prob_c

        # Shape: (n_matings × n_var)
        mask = np.random.random((n_matings, n_var)) < 0.5

        Y = np.zeros((self.n_offsprings, n_matings, n_var))

        Y[0] = np.where(mask, X[0], X[1])
        Y[1] = np.where(mask, X[1], X[0])

        Y[0, ~do_crossover] = X[0, ~do_crossover]
        Y[1, ~do_crossover] = X[1, ~do_crossover]

        return Y


class RandomReplacementMutation(Mutation):
    def __init__(self, prob_m=None):
        super().__init__()
        self.prob_m = prob_m

    def _do(self, problem, X, **kwargs):
        if self.prob_m is None:
            self.prob_m = 1.0 / problem.n_var

        # Shape: (n_individuals × n_var)
        mask = np.random.random(X.shape) < self.prob_m

        random_values = np.random.randint(problem.xl, problem.xu + 1, size=X.shape)

        return np.where(mask, random_values, X)


class StartFromZeroRepair(Repair):

    def _do(self, problem, X, **kwargs):
        I = np.where(X == 0)[1]

        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])

        return X


class RoundingRepair(Repair):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        return np.around(X).astype(int)
