import torch 
import numpy as np 
from gpytorch.kernels import Kernel
from pymoo.core.problem import Problem 

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

class CategoricalOverlap(Kernel):
    """Implementation of the categorical overlap kernel.
    This is the most basic form of the categorical kernel that essentially invokes a Kronecker delta function
    between any two elements.
    """

    has_lengthscale = True

    def __init__(self, **kwargs):
        super(CategoricalOverlap, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # First, convert one-hot to ordinal representation

        diff = x1[:, None] - x2[None, :]
        # nonzero location = different cat
        diff[torch.abs(diff) > 1e-5] = 1
        # invert, to now count same cats
        diff1 = torch.logical_not(diff).float()
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.sum(self.lengthscale * diff1, dim=-1) / torch.sum(self.lengthscale)
        else:
            # dividing by number of cat variables to keep this term in range [0,1]
            k_cat = torch.sum(diff1, dim=-1) / x1.shape[1]
        if diag:
            return torch.diag(k_cat).to(**tkwargs)
        return k_cat.to(**tkwargs)

class TransformedCategorical(CategoricalOverlap):
    """
    Second kind of transformed kernel of form:
    $$ k(x, x') = \exp(\frac{\lambda}{n}) \sum_{i=1}^n [x_i = x'_i] )$$ (if non-ARD)
    or
    $$ k(x, x') = \exp(\frac{1}{n} \sum_{i=1}^n \lambda_i [x_i = x'_i]) $$ if ARD
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp='rbf', **params):
        # print("Input shapes:", x1.shape, x2.shape)
        if x1.dim() <= 3:
            if x1.dim() == 2:
                x1 = x1.unsqueeze(0)
            if x2.dim() == 2:
                x2 = x2.unsqueeze(0)
                
            batch_size, n1, d = x1.shape
            _, n2, _ = x2.shape

            M1_expanded = x1.unsqueeze(2)  # Shape: (batch_size, n1, 1, d)
            M2_expanded = x2.unsqueeze(1)  # Shape: (batch_size, 1, n2, d)

            diff = (M1_expanded != M2_expanded).float()  # Shape: (batch_size, n1, n2, d)
            
            def rbf(d, ard):
                if ard:
                    return torch.exp(-torch.sum(d * self.lengthscale, dim=-1) / torch.sum(self.lengthscale))
                else:
                    return torch.exp(-self.lengthscale * torch.sum(d, dim=-1) / d.shape[-1])
            
            def mat52(d, ard):
                raise NotImplementedError
            
            if exp == 'rbf':
                k_cat = rbf(diff, self.ard_num_dims is not None and self.ard_num_dims > 1)
            elif exp == 'mat52':
                k_cat = mat52(diff, self.ard_num_dims is not None and self.ard_num_dims > 1)
            else:
                raise ValueError('Exponentiation scheme %s is not recognised!' % exp)
            
            if diag:
                return torch.diagonal(k_cat, dim1=1, dim2=2).squeeze(0)
            
            # print("Output shape:", k_cat.squeeze(0).shape)
            return k_cat.squeeze(0)  
        
        else:
            batch_size1, l, n1, m = x1.shape
            batch_size2, _, n2, _ = x2.shape
            
            assert batch_size2 == batch_size1

            # Expand x1 and x2 to calculate the Hamming distance
            M1_expanded = x1.unsqueeze(3)  # Shape: (batch_size, l, n1, 1, m)
            M2_expanded = x2.unsqueeze(2)  # Shape: (batch_size, l, 1, n2, m)

            # Calculate Hamming distance
            hamming_dist = (M1_expanded != M2_expanded).float().sum(dim=-1)  # Shape: (batch_size, l, n1, n2)

            def rbf(d, ard=False):
                if ard:
                    return torch.exp(-torch.sum(d / self.lengthscale, dim=-1))
                else:
                    return torch.exp(-self.lengthscale * d)

            def mat52(d):
                raise NotImplementedError

            if exp == 'rbf':
                k_cat = rbf(hamming_dist)
            elif exp == 'mat52':
                k_cat = mat52(hamming_dist)
            else:
                raise ValueError('Exponentiation scheme %s is not recognized!' % exp)

            if diag:
                return torch.diagonal(k_cat, offset=0, dim1=-2, dim2=-1).contiguous()
            print("Output shape:", k_cat.shape)
            return k_cat  # Shape: (batch_size, l, n1, n2)
        

class LCB_Problem(Problem):
    def __init__(self, n_var, n_obj, model, xl=None, xu=None):
        super().__init__(n_var=n_var, n_obj=n_obj,
                         xl = xl,
                         xu = xu,)
        self.model = model

    def _get_acq_value(self, X, model):
        X = torch.tensor(X)
        with torch.no_grad():
            X = X.to(**tkwargs)
            posterior = model.posterior(X)
            mean = posterior.mean
            var = posterior.variance
            return (mean - 0.2*(torch.sqrt(var))).detach().cpu().numpy()

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._get_acq_value(x, self.model)

class UCB_Problem(Problem):
    def __init__(self, n_var, n_obj, model, xl=None, xu=None):
        super().__init__(n_var=n_var, n_obj=n_obj,
                         xl = xl,
                         xu = xu,)
        self.model = model

    def _get_acq_value(self, X, model):
        X = torch.tensor(X)
        with torch.no_grad():
            X = X.to(**tkwargs)
            posterior = model.posterior(X)
            mean = posterior.mean
            var = posterior.variance
            return (mean + 0.2*(torch.sqrt(var))).detach().cpu().numpy()

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._get_acq_value(x, self.model) * (-1) # since we want to maximize ucb


class AcqfProblem(Problem):
    def __init__(self, n_var, acq_func, xl=None, xu=None):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
        self.acq_func = acq_func

    def _evaluate(self, x, out, *args, **kwargs):
        if isinstance(x, (np.ndarray, list)):
            x = torch.tensor(x).to(**tkwargs)
        out["F"] = self.acq_func(x.unsqueeze(1)).reshape(-1, 1).detach().cpu().numpy() * (-1)