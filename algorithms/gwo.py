import numpy as np

class GWO:
    def __init__(self, func, dim, bounds, max_fes, pop_size=30, seed=None):
        self.func     = func
        self.dim      = dim
        self.lb, self.ub = bounds
        self.max_fes  = max_fes
        self.pop_size = pop_size
        self.seed     = seed

    def optimize(self):
        rng = np.random.default_rng(self.seed)
        
        # Initialization
        pos = rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self.func(p) for p in pos])
        fes = self.pop_size

        # Identify initial leaders
        sorted_indices = np.argsort(fitness)
        alpha_pos = pos[sorted_indices[0]].copy()
        alpha_fit = fitness[sorted_indices[0]]
        beta_pos  = pos[sorted_indices[1]].copy()
        delta_pos = pos[sorted_indices[2]].copy()

        history = [alpha_fit]
        
        while fes < self.max_fes:
            # Linear decay of 'a' from 2 to 0
            a = 2.0 - (fes / self.max_fes) * 2.0
            
            for i in range(self.pop_size):
                if fes >= self.max_fes:
                    break
                
                # Vectorized update for the three leaders
                # r1, r2 are shapes (3, dim) for alpha, beta, delta
                r1 = rng.random((3, self.dim))
                r2 = rng.random((3, self.dim))
                
                A = 2 * a * r1 - a
                C = 2 * r2
                
                leaders = np.array([alpha_pos, beta_pos, delta_pos])
                
                # Calculate D_alpha, D_beta, D_delta
                D = np.abs(C * leaders - pos[i])
                # Calculate X1, X2, X3
                X_steps = leaders - A * D
                
                # Update position (average of X1, X2, X3)
                new_pos = np.clip(np.mean(X_steps, axis=0), self.lb, self.ub)
                
                new_fit = self.func(new_pos)
                fes += 1
                
                # Greedy selection (Optional: remove if you want 'Classic' GWO)
                if new_fit < fitness[i]:
                    pos[i] = new_pos
                    fitness[i] = new_fit

            # Update Alpha, Beta, and Delta
            sorted_indices = np.argsort(fitness)
            if fitness[sorted_indices[0]] < alpha_fit:
                alpha_fit = fitness[sorted_indices[0]]
                alpha_pos = pos[sorted_indices[0]].copy()
            
            beta_pos = pos[sorted_indices[1]].copy()
            delta_pos = pos[sorted_indices[2]].copy()

            history.append(alpha_fit)

        return alpha_fit, alpha_pos, history