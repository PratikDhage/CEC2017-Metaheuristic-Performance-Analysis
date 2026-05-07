import numpy as np

class TabuSearch:
    def __init__(self, func, dim, bounds, max_fes, 
                 tabu_tenure=10, n_neighbors=20, step_size=0.1, seed=None):
        self.func        = func
        self.dim         = dim
        self.lb, self.ub = bounds
        self.max_fes     = max_fes
        self.tabu_tenure = tabu_tenure
        self.n_neighbors = n_neighbors
        # Initial step size as a fraction of the range
        self.initial_step = step_size * (self.ub - self.lb)
        self.step_size   = self.initial_step
        self.seed        = seed

    def _hash(self, x, decimals=3):
        """Discretizes the continuous space for the tabu list."""
        return tuple(np.round(x, decimals))

    def optimize(self):
        rng = np.random.default_rng(self.seed)
        
        # Initial solution
        current  = rng.uniform(self.lb, self.ub, self.dim)
        curr_fit = self.func(current)
        fes      = 1

        gbest_pos = current.copy()
        gbest_fit = curr_fit

        tabu_dict = {} # stores hash -> tenure
        history   = [gbest_fit]

        while fes < self.max_fes:
            # 1. Generate Neighbourhood (Vectorized)
            perturbations = rng.normal(0, self.step_size, (self.n_neighbors, self.dim))
            candidates = np.clip(current + perturbations, self.lb, self.ub)
            
            best_cand_fit = np.inf
            best_candidate = None

            # 2. Evaluate Candidates
            for cand in candidates:
                if fes >= self.max_fes: break
                
                cand_fit = self.func(cand)
                fes += 1
                
                h = self._hash(cand)
                
                # Tabu logic + Aspiration Criterion
                is_tabu = h in tabu_dict
                if not is_tabu or cand_fit < gbest_fit:
                    if cand_fit < best_cand_fit:
                        best_cand_fit = cand_fit
                        best_candidate = cand

            # 3. Fallback: If all neighbors are tabu and none pass aspiration
            if best_candidate is None:
                best_candidate = candidates[0]
                best_cand_fit = self.func(best_candidate)
                fes += 1

            # 4. Update Tabu List
            tabu_dict[self._hash(best_candidate)] = self.tabu_tenure
            
            # Decrease tenures and remove expired
            tabu_dict = {k: v - 1 for k, v in tabu_dict.items() if v > 1}

            # 5. Move and Update Best
            if best_cand_fit < curr_fit:
                # Successful move
                current = best_candidate
                curr_fit = best_cand_fit
            else:
                # Stagnation: shrink step size to focus search
                self.step_size *= 0.99
                current = best_candidate # Move anyway to escape local optima
                curr_fit = best_cand_fit

            if curr_fit < gbest_fit:
                gbest_fit = curr_fit
                gbest_pos = current.copy()
                # Optional: Expand step size on improvement to explore further
                self.step_size = self.initial_step * 0.5 

            history.append(gbest_fit)

        return gbest_fit, gbest_pos, history