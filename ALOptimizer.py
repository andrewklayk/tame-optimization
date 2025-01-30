import torch.utils.data.dataloader
import torch.optim as optim


class ALOptimizer:
    def __init__(self,
                 net: torch.nn.Module,
                 loss_fn: callable,
                 m: int,
                 lambda_0: torch.Tensor=None,
                 aug_term: float=1.,
                 t: float=1.01,
                 constraint_fn: callable=None,
                 minimizer: str='Adam',
                 **kwargs):
        """
        Solves an inequality-constrained optimization problem using the non-smooth Augmented Lagrangian method:
        min f(x) s.t. c(x) <= 0

        Args:
            net ()
            # params (iterable): iterable of parameters to optimize.
            loss_fn (callable): the objective function.
            m (int): number of inequalities.
            lambda_0 (torch.Tensorm, optional): initial value for the Lagrange multipliers, expected to be of shape (m,).
            aug_term (float): initial value for the r (rho) parameter.
            t (float): multiplier for the r (rho) parameter.
            constraint_fn (callable, optional): the constraint function, expected: f(net) -> Tensor((m,)) 
            expected to return a tensor of constraint values.
            minimizer (str, optional): the PyTorch optimizer to minimize the objective function on each iteration; default is Adam.
            **kwargs: arguments for the minimizer.
        """
        # self._params = params
        self.net = net
        
        self._loss_fn = loss_fn
        self._constraint_fn = constraint_fn
        self._m = m
        self._lambda = lambda_0 if lambda_0 is not None else torch.ones(m)
        self._ss = aug_term
        self._t = t

        if minimizer is None or minimizer.lower() == 'adam':
            self._optimizer = optim.Adam(self.net.parameters(), **kwargs)
        elif minimizer.lower() == 'sgd':
            self._optimizer = optim.SGD(self.net.parameters(), **kwargs)
        else:
            raise ValueError(f'Unknown optimizer: {minimizer}')

        
    def optimize(self, dataloader: torch.utils.data.DataLoader, maxiter: int=3, epochs: int=3, verbose: bool=True) -> None:
        """
        Perform optimization using the Augmented Lagrangian method.

        Iteratively minimize the augmented Lagrangian function with respect to 
        the neural network parameters while updating the Lagrange multipliers and augmentation term.
        The method updates the network parameters in place and records optimization history.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing input data and labels for training.
            maxiter (int, optional): Number of outer iterations for updating the Lagrange multipliers. Default is 3.
            epochs (int, optional): Number of epochs per outer iteration for minimizing the augmented Lagrangian. Default is 3.
            verbose (bool, optional): Whether to print progress and constraint updates. Default is True.

        Returns:
            None 
        """
        self.loss_val = 0
        self.history = {'L': [], 'loss': [], 'constr': []}
        for k in range(maxiter):
            # minimize the augmented lagrangian with the current multiplier values
            for epoch in range(epochs):
                for i, data in enumerate(dataloader):
                    self._optimizer.zero_grad()
                    inputs, labels = data
                    outputs = self.net(inputs)
                    constraint_eval = self._constraint_fn(self.net)
                    loss_eval = self._loss_fn(outputs, labels)
                    
                    L = loss_eval + self._lambda @ constraint_eval + 0.5*self._ss*torch.sum(torch.square(constraint_eval))
                    L.backward()
                    self._optimizer.step()

                    ###
                    if verbose:
                        print(f'{epoch}, {i}, {loss_eval.detach().item()}, {constraint_eval.detach().item()}', end='\r')
                    ###

                    self.history['L'].append(L)
                    self.history['loss'].append(loss_eval)
                    self.history['constr'].append(constraint_eval)
                
            with torch.no_grad():
                constr = self._constraint_fn(self.net)
                if verbose:
                    print('-------')
                    print('\n')
                self._lambda += self._ss*constr
                self._ss *= self._t

        #######################
        # stopping condition? #
        #######################