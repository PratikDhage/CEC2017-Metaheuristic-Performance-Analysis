import numpy as np

class WOA:
    def __init__(self, func, dim, bounds, max_fes, pop_size=30, b=1.0, seed=None):
        self.func     = func
        self.dim      = dim
        self.lb, self.ub = bounds
        self.max_fes  = max_fes
        self.pop_size = pop_size
        self.b        = b      # Constant for defining the shape of the logarithmic spiral
        self.seed     = seed

    def optimize(self):
        rng = np.random.default_rng(self.seed)
        
        # Initialization
        pos = rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self.func(p) for p in pos])
        fes = self.pop_size

        best_idx = np.argmin(fitness)
        gbest_pos = pos[best_idx].copy()
        gbest_fit = fitness[best_idx]
        
        history = [gbest_fit]

        while fes < self.max_fes:
            # 'a' decreases linearly from 2 to 0
            a = 2.0 - (fes / self.max_fes) * 2.0
            
            for i in range(self.pop_size):
                if fes >= self.max_fes:
                    break
                
                p = rng.random()
                # l is a random number in [-1, 1]
                l = rng.uniform(-1, 1)
                
                # Parameters for encircling/searching
                # A is a vector in [-a, a]
                A = 2.0 * a * rng.random(self.dim) - a
                C = 2.0 * rng.random(self.dim)
                
                if p < 0.5:
                    # Mechanism 1: Encircling or Searching
                    if np.linalg.norm(A) < 1:
                        # Shrinking encircling (Exploitation)
                        D = np.abs(C * gbest_pos - pos[i])
                        new_pos = gbest_pos - A * D
                    else:
                        # Search for prey (Exploration)
                        # Select a random whale
                        rand_idx = rng.integers(0, self.pop_size)
                        x_rand = pos[rand_idx]
                        D = np.abs(C * x_rand - pos[i])
                        new_pos = x_rand - A * D
                else:
                    # Mechanism 2: Bubble-net attacking (Exploitation)
                    # Logarithmic spiral to simulate the bubble-net
                    dist_to_best = np.abs(gbest_pos - pos[i])
                    new_pos = dist_to_best * np.exp(self.b * l) * np.cos(2.0 * np.pi * l) + gbest_pos

                # Boundary handling and evaluation
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.func(new_pos)
                fes += 1

                # Greedy selection
                if new_fit < fitness[i]:
                    pos[i] = new_pos
                    fitness[i] = new_fit
                    
                    if new_fit < gbest_fit:
                        gbest_fit = new_fit
                        gbest_pos = new_pos.copy()

            history.append(gbest_fit)

        return gbest_fit, gbest_pos, history