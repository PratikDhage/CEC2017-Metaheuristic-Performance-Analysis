import numpy as np

class DE:
    def __init__(self, func, dim, bounds, max_fes, pop_size=50,
                 F=0.8, CR=0.9, seed=None):
        self.func     = func
        self.dim      = dim
        self.lb, self.ub = bounds
        self.max_fes  = max_fes
        self.pop_size = pop_size
        self.F        = F    # Scale factor
        self.CR       = CR   # Crossover probability
        self.seed     = seed

    def optimize(self):
        rng = np.random.default_rng(self.seed)
        
        # Initialize population
        pop = rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self.func(p) for p in pop])
        fes = self.pop_size

        gbest_idx = np.argmin(fitness)
        gbest_fit = fitness[gbest_idx]
        gbest_pos = pop[gbest_idx].copy()
        history = [gbest_fit]

        # Pre-allocate an index array for faster neighbor selection
        all_idxs = np.arange(self.pop_size)

        while fes < self.max_fes:
            for i in range(self.pop_size):
                if fes >= self.max_fes:
                    break

                # 1. Mutation (DE/rand/1)
                # Efficiently pick 3 random indices excluding 'i'
                mask = all_idxs != i
                idxs = rng.choice(all_idxs[mask], 3, replace=False)
                a, b, c = pop[idxs]
                
                mutant = a + self.F * (b - c)
                # Standard DE boundary handling: bounce back or clip
                mutant = np.clip(mutant, self.lb, self.ub)

                # 2. Binomial Crossover
                cross_pts = rng.random(self.dim) < self.CR
                # Ensure at least one component comes from the mutant
                cross_pts[rng.integers(0, self.dim)] = True
                trial = np.where(cross_pts, mutant, pop[i])

                # 3. Selection
                trial_fit = self.func(trial)
                fes += 1

                if trial_fit <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    
                    if trial_fit < gbest_fit:
                        gbest_fit = trial_fit
                        gbest_pos = trial.copy()

            history.append(gbest_fit)

        return gbest_fit, gbest_pos, history