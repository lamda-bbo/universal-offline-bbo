import numpy as np 
from typing import Optional
import torch
import time 
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model
import gpytorch
from botorch.acquisition.monte_carlo import qExpectedImprovement 
# from botorch.acquisition import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

from src.searcher.bo_utils import TransformedCategorical, UCB_Problem
from src.searcher.pymoo_utils import UniformCrossover, RandomReplacementMutation
from src.searcher.base import BaseSearcher
from src.utils import (
    RankedLogger,
)
from src.utils.io_utils import load_task_names, save_metric_to_csv, check_if_evaluated

log = RankedLogger(__name__, rank_zero_only=True)

class BOqLogEISearcher(BaseSearcher):
    def __init__(
        self,
        noise_se: float,
        batch_size: int,
        num_restarts: int, 
        raw_samples: int,
        batch_limit: int, 
        maxiter: int,
        iterations: int,
        mc_samples: int,
        gp_samples: int,
        MAXIMIZE: bool = True,
        EVAL_STABILITY: bool = False,
        *args,
        **kwargs
    ) -> None:
        super(BOqLogEISearcher, self).__init__(*args, **kwargs)
        self.noise_se = noise_se 
        self.batch_size = batch_size
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.batch_limit = batch_limit
        self.maxiter = maxiter
        self.iterations = iterations
        self.mc_samples = mc_samples
        self.gp_samples = gp_samples
        self.EVAL_STABILITY = EVAL_STABILITY
        self.MAXIMIZE = MAXIMIZE
        self.sol_collection = []
        self.y_collection = []
    
    def run(self) -> np.ndarray:
        tkwargs = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float64
        }
        
        if self.task.task_type == "Categorical":
            self.gp_samples = 100

        initial_x, initial_y = self.get_initial_designs(
            x=self.task.x_np, y=self.task.y_np, k=self.gp_samples
        )
        initial_x = torch.from_numpy(np.array(initial_x)).to(**tkwargs)
        initial_y = torch.from_numpy(np.array(initial_y)).to(**tkwargs)
        input_size = initial_x.shape[1]
            
        xl, xu = self.task.bounds

        def objective(x):
            # print(x.shape, self.score_fn(x).shape)
            fs = self.score_fn(x)
            
            if not self.MAXIMIZE:
                fs = fs * (-1)
            return fs

        if self.task.task_type == "Continuous":
            
            NOISE_SE = self.noise_se
            train_yvar = torch.tensor(NOISE_SE ** 2).to(**tkwargs)

            bounds = torch.tensor(
                [xl,
                xu]
            ).to(**tkwargs)
            
            standard_bounds = torch.zeros_like(bounds)
            standard_bounds[1] = 1
            
            def initialize_model(train_x, train_obj, state_dict=None):
                train_x = (train_x - bounds[0]) / (bounds[1] - bounds[0])
                train_obj = (train_obj - train_obj.mean()) / train_obj.std()
                # define models for objective
                model_obj = FixedNoiseGP(train_x, train_obj,
                                        train_yvar.expand_as(train_obj)).to(train_x)
                # combine into a multi-output GP model
                model = ModelListGP(model_obj)
                mll = SumMarginalLogLikelihood(model.likelihood, model)
                # load state dict if it is passed
                if state_dict is not None:
                    model.load_state_dict(state_dict)
                return mll, model

            def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
                return Z[..., 0]

            # define a feasibility-weighted objective for optimization
            obj = GenericMCObjective(obj_callable)

            BATCH_SIZE = self.batch_size
             
            
            def optimize_acqf_and_get_observation(acq_func):
                try:
                    candidates_normalized, _ = optimize_acqf(
                        acq_function=acq_func,
                        bounds=standard_bounds,
                        q=BATCH_SIZE,
                        num_restarts=self.num_restarts,
                        raw_samples=self.raw_samples,
                        options={"batch_limit": self.batch_limit, "maxiter": self.maxiter}
                    )

                    candidates = candidates_normalized * (bounds[1] - bounds[0]) + bounds[0]

                    exact_obj = torch.from_numpy(
                        np.array(objective(candidates.detach().cpu().numpy()))
                    ).to(**tkwargs)
                    
                    new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
                    return candidates, new_obj 

                except RuntimeError:
                    return None

            N_BATCH = self.iterations
            MC_SAMPLES = self.mc_samples

            best_observed_ei = []

            # call helper functions to generate initial training data and initialize model
            train_x_ei = initial_x.reshape([initial_x.shape[0], input_size])
            train_x_ei = torch.tensor(train_x_ei).to(**tkwargs)

            train_obj_ei = initial_y.reshape([initial_y.shape[0], 1])
            train_obj_ei = torch.tensor(train_obj_ei).to(**tkwargs)

            x_search = torch.zeros(0, input_size).to(**tkwargs)
            y_search = torch.zeros(0, 1).to(**tkwargs)

            best_observed_value_ei = train_obj_ei.max().item()
            mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei)
            best_observed_ei.append(best_observed_value_ei)

            # run N_BATCH rounds of BayesOpt after the initial random batch
            for iteration in range(1, N_BATCH + 1):

                t0 = time.time()

                # fit the models
                fit_gpytorch_model(mll_ei)

                # define the qEI acquisition module using a QMC sampler
                qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

                # for best_f, we use the best observed noisy values as an approximation
                tmp_train_obj = (train_obj_ei - train_obj_ei.mean()) / train_obj_ei.std()
                qLogEI = qExpectedImprovement(
                    model=model_ei, best_f=tmp_train_obj.max(),
                    sampler=qmc_sampler, objective=obj)

                # optimize and get new observation
                result = optimize_acqf_and_get_observation(qLogEI)
                if result is None:
                    print("RuntimeError was encountered, most likely a "
                        "'symeig_cpu: the algorithm failed to converge'")
                    break
                new_x_ei, new_obj_ei = result
                self.sol_collection.append(new_x_ei.detach().cpu().numpy())
                self.y_collection.extend(new_obj_ei.flatten().tolist())
                # update training points
                train_x_ei = torch.cat([train_x_ei, new_x_ei])
                train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

                x_search = torch.cat([x_search, new_x_ei])
                y_search = torch.cat([y_search, new_obj_ei])

                # update progress
                best_value_ei = obj(train_x_ei).max().item()
                best_observed_ei.append(best_value_ei)

                # reinitialize the models so they are ready for fitting on next iteration
                # use the current state dict to speed up fitting
                mll_ei, model_ei = initialize_model(
                    train_x_ei, train_obj_ei, model_ei.state_dict())

                t1 = time.time()
                log.info(f"Batch {iteration:>2}: best_value = "
                    f"({best_value_ei:>4.2f}), "
                    f"time = {t1 - t0:>4.2f}.")

            x_sol = x_search.detach().cpu().numpy()
            y_sol = y_search.detach().cpu().numpy()
            
            solution = x_sol[np.argsort(y_sol.squeeze())[-self.num_solutions:]]
            return solution
        
        elif self.task.task_type == "Categorical":
            def _get_model(train_X: torch.Tensor, train_Y: torch.Tensor):
                # assert 0, (train_X.shape, train_Y.shape)
                Y_var = torch.full_like(train_Y, 0.01).to(**tkwargs)
                kernel = TransformedCategorical().to(**tkwargs)
                model_obj = FixedNoiseGP(train_X, train_Y, Y_var, covar_module=kernel).to(**tkwargs)
                model = ModelListGP(model_obj).to(**tkwargs)
                mll = SumMarginalLogLikelihood(model.likelihood, model)
                # likelihood = gpytorch.likelihoods.GaussianLikelihood().to(**tkwargs)
                # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(**tkwargs)
                fit_gpytorch_model(mll)
                return model
            
            train_x_ei = initial_x.reshape([initial_x.shape[0], input_size])
            train_x_ei = torch.tensor(train_x_ei).to(**tkwargs)

            train_obj_ei = initial_y.reshape([initial_y.shape[0], 1])
            train_obj_ei = torch.tensor(train_obj_ei).to(**tkwargs)

            N_BATCH = self.iterations

            for iteration in range(1, N_BATCH + 1):
                t0 = time.time()
                model = _get_model(train_x_ei, train_obj_ei).to(**tkwargs)
                problem = UCB_Problem(input_size, 1, model, xl, xu)
                operator = {
                    "crossover": UniformCrossover(),
                    "mutation": RandomReplacementMutation(),
                }
                _algo = GA(pop_size=32, **operator, eliminate_duplicates=True,
                           sampling=train_x_ei[torch.argsort(train_obj_ei.flatten())[-128:]].squeeze().detach().cpu().numpy()
                           )
                res = minimize(problem=problem, algorithm=_algo, termination=('n_gen', 200), verbose=False)
                x = res.pop.get('X')
                print(x)
                y = objective(x)
                print(y.max(), self.task.evaluate(x.astype(np.int64),
                                                  return_normalized_y=True)['normalized-score-100th'])

                train_x_ei = torch.cat([train_x_ei, torch.from_numpy(x).to(train_x_ei)])
                train_obj_ei = torch.cat([train_obj_ei, torch.from_numpy(y).to(train_obj_ei)])

                best_value_ei = train_obj_ei.max().item()

                t1 = time.time()
                log.info(f"Batch {iteration:>2}: best_value = "
                    f"({best_value_ei:>4.2f}), "
                    f"time = {t1 - t0:>4.2f}.")

            x_sol = train_x_ei.detach().cpu().numpy()
            y_sol = train_obj_ei.detach().cpu().numpy()
            
            solution = x_sol[np.argsort(y_sol.squeeze())[-self.num_solutions:]]
            # solution = x_sol
            return solution.astype(np.int64)