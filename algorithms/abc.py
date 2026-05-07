import numpy as np

class ABC:
    def __init__(self, func, dim, bounds, max_fes, colony_size=50, limit=None, seed=None):
        self.func        = func
        self.dim         = dim
        self.lb, self.ub = bounds
        self.max_fes     = max_fes
        self.n_employed  = colony_size // 2
        self.n_onlooker  = colony_size // 2
        # Classic limit: SN * D (where SN is number of employed bees)
        self.limit       = limit if limit else self.n_employed * self.dim
        self.seed        = seed

    def _fitness(self, val):
        """Standard ABC fitness mapping."""
        return 1.0 / (1.0 + val) if val >= 0 else 1.0 + abs(val)

    def optimize(self):
        rng = np.random.default_rng(self.seed)
        
        # Initialization
        pos   = rng.uniform(self.lb, self.ub, (self.n_employed, self.dim))
        obj   = np.array([self.func(p) for p in pos])
        trial = np.zeros(self.n_employed, dtype=int)
        fes   = self.n_employed

        gbest_fit = np.min(obj)
        gbest_pos = pos[np.argmin(obj)].copy()
        history   = [gbest_fit]

        while fes < self.max_fes:
            # --- Employed Bee Phase ---
            for i in range(self.n_employed):
                if fes >= self.max_fes: break
                
                # Pick a random dimension and a random neighbor k != i
                j = rng.integers(0, self.dim)
                k = rng.integers(0, self.n_employed)
                while k == i: k = rng.integers(0, self.n_employed)

                phi = rng.uniform(-1, 1)
                candidate = pos[i].copy()
                candidate[j] = np.clip(pos[i, j] + phi * (pos[i, j] - pos[k, j]), self.lb, self.ub)
                
                cand_obj = self.func(candidate)
                fes += 1
                
                if cand_obj < obj[i]:
                    pos[i] = candidate
                    obj[i] = cand_obj
                    trial[i] = 0
                else:
                    trial[i] += 1

            # --- Onlooker Bee Phase ---
            fit = np.array([self._fitness(o) for o in obj])
            total_fit = np.sum(fit)
            
            # --- Safety Check: Prevent division by zero or NaN probabilities ---
            if total_fit == 0 or np.isnan(total_fit):
                probs = np.full(self.n_employed, 1.0 / self.n_employed)
            else:
                probs = fit / total_fit
            
            for _ in range(self.n_onlooker):
                if fes >= self.max_fes: break
                
                # Select source based on probability (Roulette Wheel)
                i = rng.choice(self.n_employed, p=probs)
                
                j = rng.integers(0, self.dim)
                k = rng.integers(0, self.n_employed)
                while k == i: k = rng.integers(0, self.n_employed)

                phi = rng.uniform(-1, 1)
                candidate = pos[i].copy()
                candidate[j] = np.clip(pos[i, j] + phi * (pos[i, j] - pos[k, j]), self.lb, self.ub)
                
                cand_obj = self.func(candidate)
                fes += 1
                
                if cand_obj < obj[i]:
                    pos[i] = candidate
                    obj[i] = cand_obj
                    trial[i] = 0
                else:
                    trial[i] += 1

            # --- Scout Bee Phase ---
            if np.max(trial) > self.limit:
                if fes < self.max_fes:
                    i = np.argmax(trial)
                    pos[i] = rng.uniform(self.lb, self.ub, self.dim)
                    obj[i] = self.func(pos[i])
                    fes += 1
                    trial[i] = 0

            # Update Global Best
            current_min_idx = np.argmin(obj)
            if obj[current_min_idx] < gbest_fit:
                gbest_fit = obj[current_min_idx]
                gbest_pos = pos[current_min_idx].copy()

            history.append(gbest_fit)
            
            if fes >= self.max_fes: break

        return gbest_fit, gbest_pos, history