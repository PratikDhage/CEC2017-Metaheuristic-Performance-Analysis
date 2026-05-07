import numpy as np

class PSO:
    def __init__(self, func, dim, bounds, max_fes, pop_size=30, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5, seed=None):
        self.func     = func
        self.dim      = dim
        self.lb, self.ub = bounds
        self.max_fes  = max_fes
        self.pop_size = pop_size
        self.w_max, self.w_min = w_max, w_min 
        self.c1, self.c2 = c1, c2
        self.seed     = seed
        # Velocity clamping limit
        self.v_limit  = 0.2 * (self.ub - self.lb)

    def optimize(self):
        rng = np.random.default_rng(self.seed)
        
        pos = rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        vel = rng.uniform(-self.v_limit, self.v_limit, (self.pop_size, self.dim))

        fitness   = np.array([self.func(p) for p in pos])
        fes       = self.pop_size
        pbest_pos = pos.copy()
        pbest_fit = fitness.copy()

        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        history = [gbest_fit]

        while fes < self.max_fes:
            # 1. Update Inertia Weight (Linear Decay)
            w = self.w_max - (fes / self.max_fes) * (self.w_max - self.w_min)
            
            r1 = rng.random((self.pop_size, self.dim))
            r2 = rng.random((self.pop_size, self.dim))

            # 2. Update Velocity with Clamping
            vel = (w * vel
                   + self.c1 * r1 * (pbest_pos - pos)
                   + self.c2 * r2 * (gbest_pos - pos))
            vel = np.clip(vel, -self.v_limit, self.v_limit)

            # 3. Update Position
            pos = np.clip(pos + vel, self.lb, self.ub)

            # 4. Evaluation
            fitness = np.array([self.func(p) for p in pos])
            fes    += self.pop_size

            # 5. Update personal and global bests
            improved = fitness < pbest_fit
            pbest_pos[improved] = pos[improved]
            pbest_fit[improved] = fitness[improved]

            if np.min(pbest_fit) < gbest_fit:
                gbest_idx = np.argmin(pbest_fit)
                gbest_fit = pbest_fit[gbest_idx]
                gbest_pos = pbest_pos[gbest_idx].copy()

            history.append(gbest_fit)

        return gbest_fit, gbest_pos, history