import numpy as np

class GA:
    def __init__(self, func, dim, bounds, max_fes, pop_size=50,
                 pc=0.9, pm=None, eta_c=20, eta_m=20, seed=None):
        self.func     = func
        self.dim      = dim
        self.lb, self.ub = bounds
        self.max_fes  = max_fes
        self.pop_size = pop_size
        self.pc       = pc
        self.pm       = pm if pm else 1.0 / dim
        self.eta_c    = eta_c
        self.eta_m    = eta_m
        self.seed     = seed

    def _sbx_crossover(self, rng, p1, p2):
        u = rng.random(self.dim)
        beta = np.where(u <= 0.5,
                        (2 * u) ** (1 / (self.eta_c + 1)),
                        (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1)))
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return np.clip(c1, self.lb, self.ub), np.clip(c2, self.lb, self.ub)

    def _poly_mutation(self, rng, x):
        u = rng.random(self.dim)
        delta = np.where(u < 0.5,
                         (2 * u) ** (1 / (self.eta_m + 1)) - 1,
                         1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1)))
        mask = rng.random(self.dim) < self.pm
        x = x + mask * delta * (self.ub - self.lb)
        return np.clip(x, self.lb, self.ub)

    def _tournament(self, rng, fitness, k=3):
        idx = rng.choice(len(fitness), k, replace=False)
        return idx[np.argmin(fitness[idx])]

    def optimize(self):
        rng = np.random.default_rng(self.seed)
        
        # Initial Population
        pop = rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self.func(p) for p in pop])
        fes = self.pop_size

        gbest_idx = np.argmin(fitness)
        gbest_fit = fitness[gbest_idx]
        gbest_pos = pop[gbest_idx].copy()
        history = [gbest_fit]

        while fes < self.max_fes:
            offspring = []
            
            # Generate offspring
            while len(offspring) < self.pop_size:
                p1 = pop[self._tournament(rng, fitness)]
                p2 = pop[self._tournament(rng, fitness)]
                
                if rng.random() < self.pc:
                    c1, c2 = self._sbx_crossover(rng, p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                offspring.append(self._poly_mutation(rng, c1))
                if len(offspring) < self.pop_size:
                    offspring.append(self._poly_mutation(rng, c2))

            # Budget-aware evaluation
            offspring_fit = []
            for child in offspring:
                if fes < self.max_fes:
                    offspring_fit.append(self.func(child))
                    fes += 1
                else:
                    # If out of budget, assign infinity so they are ignored in selection
                    offspring_fit.append(np.inf)

            offspring_fit = np.array(offspring_fit)
            offspring = np.array(offspring)

            # (mu + lambda) selection
            combined_pop = np.vstack([pop, offspring])
            combined_fit = np.hstack([fitness, offspring_fit])
            
            # Sort and pick best N
            best_indices = np.argsort(combined_fit)[:self.pop_size]
            pop = combined_pop[best_indices]
            fitness = combined_fit[best_indices]

            # Update Global Best
            if fitness[0] < gbest_fit:
                gbest_fit = fitness[0]
                gbest_pos = pop[0].copy()

            history.append(gbest_fit)

        return gbest_fit, gbest_pos, history