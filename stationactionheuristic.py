import numpy as np

class HeuristicPolicyValueIteration:
    """
    Compute the value function V(B) under a fixed-threshold heuristic policy
    via value iteration solving:

      V(B) = (θ + β)^(-1) [ h(B)
                           + μ·P^H[V](B)
                           + ∑_{i=1}^n (λ_i·A_i^{H0}[V](B)
                                        + ω_i·A_i^{H1}[V](B)
                                        + ω_i·A_i^{H2}[V](B) ) ]

    Operators fix thresholds b0* and b_i*:
      P^H[V](B)   = V(B + (b0* - b0)e0) if b0 < b0*, else V(B)
      A_i^{H0}[V](B) = V(B + e_i) if b_i < b_i*, else V(B) + c_i^r
      A_i^{H1}[V](B) = V(B - e0 - e_i) if b0>0 and b_i>0, else V(B) + c_i^d
      A_i^{H2}[V](B) = V(B - e_i)    if b_i>0, else V(B)
    """
    def __init__(
        self,
        buffercaps,
        hold_cost,
        reject_costs,
        loss_costs,
        mu,
        arrival_rates,
        due_rates,
        theta
    ):
        # capacities [B0, B1...Bn]
        self.buffercaps = np.array(buffercaps, dtype=int)
        self.hold_cost = float(hold_cost)
        self.reject_costs = np.array(reject_costs, dtype=float)
        self.loss_costs = np.array(loss_costs, dtype=float)
        self.mu = float(mu)
        self.lmd = np.array(arrival_rates, dtype=float)
        self.omg = np.array(due_rates, dtype=float)
        self.theta = float(theta)

        # state space
        self.n = len(self.buffercaps)
        self.num_states = int(np.prod(self.buffercaps + 1))
        self.beta = self.mu + float(self.lmd.sum() + self.omg.sum())

    def idx_to_state(self, idx):
        B = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            B[i] = idx % (self.buffercaps[i] + 1)
            idx //= (self.buffercaps[i] + 1)
        return B

    def state_to_idx(self, B):
        idx = 0; mul = 1
        for i in range(self.n):
            idx += B[i] * mul
            mul *= (self.buffercaps[i] + 1)
        return idx

    def evaluate_policy(self, b0_star, b_star, tol=1e-6, max_iter=5000):
        """
        Perform value iteration for a given set of thresholds.
        Returns the converged value function array.
        """
        V = np.zeros(self.num_states, dtype=float)
        for it in range(1, max_iter+1):
            V_old = V.copy()
            newV = np.zeros_like(V)
            for idx in range(self.num_states):
                B = self.idx_to_state(idx)
                b0 = B[0]
                # holding cost
                h = self.hold_cost * b0
                # P^H
                if b0 < b0_star:
                    Bp = B.copy(); Bp[0] = b0_star
                    phv = V_old[self.state_to_idx(Bp)]
                else:
                    phv = V_old[idx]
                ev_sum = 0.0
                for i in range(1, self.n):
                    bi = B[i]
                    # arrival
                    if bi < b_star[i-1]:
                        Ba = B.copy(); Ba[i] += 1
                        v_arr = V_old[self.state_to_idx(Ba)]
                    else:
                        v_arr = V_old[idx] + self.reject_costs[i-1]
                    # due
                    if b0 > 0 and bi > 0:
                        Bd = B.copy(); Bd[0] -= 1; Bd[i] -= 1
                        v_due = V_old[self.state_to_idx(Bd)]
                    else:
                        v_due = V_old[idx] + self.loss_costs[i-1]
                    # abandonment
                    if bi > 0:
                        Bc = B.copy(); Bc[i] -= 1
                        v_abn = V_old[self.state_to_idx(Bc)]
                    else:
                        v_abn = V_old[idx]
                    ev_sum += self.lmd[i-1] * v_arr + self.omg[i-1] * (v_due + v_abn)
                newV[idx] = (h + self.mu * phv + ev_sum) / (self.theta + self.beta)
            delta = np.max(np.abs(newV - V_old))
            V = newV
            if delta < tol:
                break
        return V

    def find_best_thresholds(self):
        """
        Brute-force search over all threshold combinations to minimize V at zero state.
        Returns best (b0_star, b_star_list, V) tuple.
        """
        B0_cap = self.buffercaps[0]
        prod_combinations = []
        best_score = np.inf
        best_params = None

        # generate all combinations
        # b0_star from 0..B0_cap
        # each b_i* from 0..buffercaps[i]
        backlog_caps = self.buffercaps[1:]
        from itertools import product
        for b0_star in range(B0_cap+1):
            for b_star in product(*[range(cap+1) for cap in backlog_caps]):
                V = self.evaluate_policy(b0_star, list(b_star))
                score = V[self.state_to_idx(np.zeros(self.n, int))]
                if score < best_score:
                    best_score = score
                    best_params = (b0_star, list(b_star), V)
        return best_params

# Example:
if __name__ == '__main__':
    caps = [8, 5, 5]
    hv = HeuristicPolicyValueIteration(
        buffercaps=caps,
        hold_cost=2,
        reject_costs=[10,12],
        loss_costs=[5,6],
        mu=0.6,
        arrival_rates=[0.4,0.1],
        due_rates=[0.401,0.101],
        theta=0.05
    )
    b0_star, b_star, V = hv.find_best_thresholds()
    print(f"Optimal thresholds: b0*={b0_star}, b*={b_star}")
    # V now holds the value for the best heuristic policy
