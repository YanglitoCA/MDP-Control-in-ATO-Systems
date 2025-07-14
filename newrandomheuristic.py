import numpy as np
import csv
from itertools import product

# ======= ThresholdHeuristicMDP and support functions (same as your code, but fixed for 4 products) =======

class ThresholdHeuristicMDP:
    def __init__(self, prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta,
                 b0_star=None, bi_star=None):
        self.prod_cap = prod_cap
        self.buffercap = np.concatenate(([prod_cap], buf_caps))
        self.invc = invc
        self.rejoc = rejoc
        self.unsatdc = unsatdc
        self.mu = mu
        self.lmd = lmd
        self.omg = omg
        self.theta = theta
        self.n = len(self.buffercap)
        self.state_size = np.prod(self.buffercap + 1)
        self.V = np.zeros(self.state_size)
        # --- Thresholds ---
        self.b0_star = self.prod_cap if b0_star is None else b0_star
        self.bi_star = buf_caps if bi_star is None else bi_star
        self.beta = self.mu + sum(self.lmd[i] + self.buffercap[i+1] * self.omg[i]
                                  for i in range(len(self.lmd)))

    def state_to_number(self, inputstate):
        sn = 0
        prestatenumber = 1
        for i in range(self.n):
            sn += inputstate[i] * prestatenumber
            if i < self.n - 1:
                prestatenumber *= (self.buffercap[i] + 1)
        return int(sn)

    def number_to_state(self, inputstatenumber):
        remaining = int(inputstatenumber)
        statemultiplier = 1
        systemstate = np.zeros(self.n, dtype=int)
        for i in range(self.n - 1):
            statemultiplier *= (self.buffercap[i] + 1)
        for i in range(self.n):
            systemstate[self.n - i - 1] = remaining // statemultiplier
            remaining %= statemultiplier
            if i < self.n - 1:
                statemultiplier //= (self.buffercap[self.n - 2 - i] + 1)
        return systemstate

    def control_decision(self, state, b0_star=None, bi_star=None):
        """Threshold-based policy."""
        if b0_star is None:
            b0_star = self.b0_star
        if bi_star is None:
            bi_star = self.bi_star

        control = []
        # Production
        control.append(1 if state[0] < b0_star else 0)
        # Order acceptance for each product
        for i in range(1, self.n):
            control.append(1 if state[i] < bi_star[i-1] else 0)
        # Fulfillment for each product
        for i in range(1, self.n):
            control.append(1 if state[0] > 0 and state[i] > 0 else 0)
        return control

    def state_trans_value_heuristic(self, state):
        """Bellman operator for the threshold heuristic policy."""
        b0 = state[0]
        n = self.n - 1
        h = self.invc * b0
        beta = self.beta

        ctrl = self.control_decision(state)
        svalue = h  # immediate cost

        # Production operator P^H V
        if b0 < self.b0_star:
            state_p = state.copy()
            state_p[0] = self.b0_star
            svalue += self.mu * self.V[self.state_to_number(state_p)]
        else:
            svalue += self.mu * self.V[self.state_to_number(state)]

        for i in range(1, self.n):
            # --- Order acceptance operator A_i^H0 V
            if state[i] < self.bi_star[i-1]:
                state_a = state.copy()
                state_a[i] += 1
                svalue += self.lmd[i-1] * self.V[self.state_to_number(state_a)]
            else:
                svalue += self.lmd[i-1] * (self.V[self.state_to_number(state)] + self.rejoc[i-1])

            # --- Fulfillment operator A_i^H1 V
            if state[0] > 0:
                if state[i] > 0:
                    state_f = state.copy()
                    state_f[0] -= 1
                    state_f[i] -= 1
                    svalue += self.omg[i-1] * state[i] * self.V[self.state_to_number(state_f)]
                else:
                    svalue += self.omg[i-1] * state[i] * (self.V[self.state_to_number(state)] + self.unsatdc[i-1])
            else:
                svalue += self.omg[i-1] * state[i] * (self.V[self.state_to_number(state)] + self.unsatdc[i-1])

            # --- Unsatisfied operator A_i^H2 V
            svalue += self.omg[i-1] * (self.buffercap[i] - state[i]) * self.V[self.state_to_number(state)]

        svalue /= (beta + self.theta)
        return svalue

    def value_iteration_heuristic(self, max_iter=1000, tol=1e-4, verbose=False):
        for iteration in range(max_iter):
            V_new = np.zeros_like(self.V)
            delta = 0
            for sn in range(self.state_size):
                state = self.number_to_state(sn)
                v = self.state_trans_value_heuristic(state)
                V_new[sn] = v
                delta = max(delta, abs(v - self.V[sn]))
            self.V = V_new
            if verbose:
                print(f"Iteration {iteration}, max change {delta:.6f}")
            if delta < tol:
                break

    def print_policy_thresholds(self):
        print(f"Production threshold b0*: {self.b0_star}")
        print(f"Order acceptance thresholds bi*: {self.bi_star}")

def simulate_system_threshold(
    solver, init_state, sim_time=100_000, random_seed=0,
    price=None, print_summary=True, csv_filename=None
):
    """Simulate the system using the threshold heuristic policy."""
    np.random.seed(random_seed)
    n = solver.n - 1
    state = np.array(init_state, dtype=int)
    total_cost = 0.0
    total_profit = 0.0
    total_orders = np.zeros(n, int)
    total_accepted = np.zeros(n, int)
    invc = solver.invc
    rejoc = solver.rejoc
    unsatdc = solver.unsatdc
    lmd = solver.lmd
    omg = solver.omg
    mu = solver.mu

    if price is None:
        price = np.zeros(n)
    avg_lead = mu

    if csv_filename:
        csvfile = open(csv_filename, 'w', newline='')
        writer = csv.writer(csvfile)
        header = ['t'] + [f'state_{i}' for i in range(len(state))] + [f'ctrl_{i}' for i in range(1+2*n)]
        writer.writerow(header)
    else:
        writer = None

    for t in range(sim_time):
        ctrl = solver.control_decision(state)

        if writer:
            writer.writerow([t] + state.tolist() + list(ctrl))

        # --- Arrivals and acceptance ---
        for j in range(n):
            accept_idx = 1 + j
            due_idx    = 1 + n + j

            # Arrival
            if np.random.rand() < lmd[j]:
                total_orders[j] += 1
                if ctrl[accept_idx] == 1 and state[j+1] < solver.buffercap[j+1]:
                    state[j+1] += 1
                    total_accepted[j] += 1
                else:
                    total_cost += rejoc[j]
            # Due
            if state[j+1] > 0 and np.random.rand() < omg[j]*state[j+1]:
                if ctrl[due_idx] == 1 and state[0] > 0:
                    state[0] -= 1
                    state[j+1] -= 1
                    total_profit += price[j]
                else:
                    total_cost += unsatdc[j]

        # --- Replenishment (production) ---
        if np.random.rand() < avg_lead and state[0] < solver.b0_star:
            state[0] = solver.b0_star

        # --- Holding cost for production buffer ---
        total_cost += state[0] * invc

        # Boundaries
        for k in range(len(state)):
            state[k] = max(0, min(state[k], solver.buffercap[k]))

    if writer:
        csvfile.close()

    avg_cost = total_cost / sim_time
    results = {
        'Total Cost': total_cost,
        'Total Profit': total_profit,
        'Total Orders': total_orders.copy(),
        'Total Accepted': total_accepted.copy(),
        'Average Cost Per Time': avg_cost,
    }
    if print_summary:
        print("\n=== Simulation Results ===")
        print(f"Total Cost:           {total_cost:.2f}")
        print(f"Total Profit:         {total_profit:.2f}")
        print(f"Total Orders:         {total_orders}")
        print(f"Total Accepted Orders:{total_accepted}")
        print(f"Average cost per time unit: {avg_cost:.4f}")
        if csv_filename:
            print(f"Simulation logged to: {csv_filename}")
    return results

def optimize_thresholds_value_iteration(
    prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta,
    search_ranges=None, sim_time=10000, price=None, verbose=False,
    max_iter=10
):
    # Setup allowed threshold ranges if not given
    if search_ranges is None:
        search_ranges = {
            'b0_star': range(prod_cap+1),
            'bi_star': [range(b+1) for b in buf_caps]
        }
    # Start from the middle
    b0_star = prod_cap // 2
    bi_star = np.array([b//2 for b in buf_caps])
    best_cost = float('inf')
    best_b0 = b0_star
    best_bi = bi_star.copy()

    for iteration in range(max_iter):
        improved = False
        # Policy Evaluation
        solver = ThresholdHeuristicMDP(prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta,
                                       b0_star=b0_star, bi_star=bi_star)
        solver.value_iteration_heuristic(verbose=False)
        avg_cost = simulate_system_threshold(
            solver,
            init_state=np.zeros(len(buf_caps)+1, dtype=int),
            sim_time=sim_time,
            random_seed=123,
            price=price,
            print_summary=False
        )['Average Cost Per Time']
        if verbose:
            print(f"Iteration {iteration}: thresholds b0*={b0_star}, bi*={bi_star}, cost={avg_cost:.3f}")
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_b0 = b0_star
            best_bi = bi_star.copy()

        # --- Policy Improvement for b0* ---
        for new_b0 in search_ranges['b0_star']:
            if new_b0 == b0_star: continue
            solver = ThresholdHeuristicMDP(prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta,
                                           b0_star=new_b0, bi_star=bi_star)
            solver.value_iteration_heuristic(verbose=False)
            avg_cost_new = simulate_system_threshold(
                solver,
                init_state=np.zeros(len(buf_caps)+1, dtype=int),
                sim_time=sim_time,
                random_seed=123,
                price=price,
                print_summary=False
            )['Average Cost Per Time']
            if avg_cost_new < best_cost:
                if verbose:
                    print(f"Improved b0*: {b0_star} → {new_b0}, cost: {avg_cost_new:.3f}")
                b0_star = new_b0
                best_cost = avg_cost_new
                best_b0 = new_b0
                best_bi = bi_star.copy()
                improved = True

        # --- Policy Improvement for each bi* ---
        for i in range(len(bi_star)):
            for new_bi in search_ranges['bi_star'][i]:
                if new_bi == bi_star[i]: continue
                bi_try = bi_star.copy()
                bi_try[i] = new_bi
                solver = ThresholdHeuristicMDP(prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta,
                                               b0_star=b0_star, bi_star=bi_try)
                solver.value_iteration_heuristic(verbose=False)
                avg_cost_new = simulate_system_threshold(
                    solver,
                    init_state=np.zeros(len(buf_caps)+1, dtype=int),
                    sim_time=sim_time,
                    random_seed=123,
                    price=price,
                    print_summary=False
                )['Average Cost Per Time']
                if avg_cost_new < best_cost:
                    if verbose:
                        print(f"Improved bi*[{i}]: {bi_star[i]} → {new_bi}, cost: {avg_cost_new:.3f}")
                    bi_star = bi_try
                    best_cost = avg_cost_new
                    best_b0 = b0_star
                    best_bi = bi_star.copy()
                    improved = True
        if not improved:
            break
    return best_b0, best_bi, best_cost

# ======= Main Experiment: Generate 5 random 4-product systems, optimize and simulate each =======

def random_system_params(n_products=4):
    prod_cap = np.random.randint(3, 8)                       # 3 to 7
    buf_caps = np.random.randint(2, 6, size=n_products)      # 2 to 5 per product
    invc = np.random.uniform(0.5, 2.0)                       # 0.5 to 2.0
    rejoc = np.random.randint(5, 20, size=n_products)        # 5 to 19 per product
    unsatdc = np.random.randint(2, 10, size=n_products)      # 2 to 9 per product
    mu = np.random.uniform(0.6, 0.95)                        # 0.6 to 0.95
    lmd = np.random.uniform(0.1, 0.3, size=n_products)       # 0.1 to 0.3 per product
    omg = np.random.uniform(0.15, 0.25, size=n_products)     # 0.15 to 0.25 per product
    theta = np.random.uniform(0.02, 0.09)                    # 0.02 to 0.09
    price = np.random.randint(80, 150, size=n_products)      # 80 to 149 per product
    init_state = np.random.randint(0, prod_cap+1, size=n_products+1)
    return prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta, price, init_state

if __name__ == "__main__":
    output_csv = 'system_simulation_results.csv'
    systems = []
    # Prepare CSV header
    header = [
        'SystemIndex',
        'prod_cap',
        'buf_caps',
        'invc',
        'rejoc',
        'unsatdc',
        'mu',
        'lmd',
        'omg',
        'theta',
        'price',
        'init_state',
        'b0_star_opt',
        'bi_star_opt',
        'avg_cost',
        'TotalCost',
        'TotalProfit',
        'TotalOrders',
        'TotalAccepted'
    ]

    results_rows = []
    for i in range(5):
        # Generate random parameters
        prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta, price, init_state = random_system_params(4)
        print(f"\n==== SYSTEM #{i+1} PARAMETERS ====")
        print(f"prod_cap={prod_cap}")
        print(f"buf_caps={buf_caps}")
        print(f"invc={invc}")
        print(f"rejoc={rejoc}")
        print(f"unsatdc={unsatdc}")
        print(f"mu={mu}")
        print(f"lmd={lmd}")
        print(f"omg={omg}")
        print(f"theta={theta}")
        print(f"price={price}")
        print(f"init_state={init_state}")

        search_ranges = {
            'b0_star': range(prod_cap+1),
            'bi_star': [range(0, b+1) for b in buf_caps]
        }
        # Optimize thresholds
        best_b0, best_bi, best_cost = optimize_thresholds_value_iteration(
            prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta,
            search_ranges=search_ranges,
            sim_time=3000,
            price=price,
            verbose=False,
            max_iter=5
        )
        print(f"Best thresholds: b0*={best_b0}, bi*={best_bi}, avg_cost={best_cost:.4f}")

        solver = ThresholdHeuristicMDP(prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta,
                                       b0_star=best_b0, bi_star=best_bi)
        solver.value_iteration_heuristic(verbose=False)

        # Simulate
        sim_result = simulate_system_threshold(
            solver,
            init_state=init_state,
            sim_time=15000,
            random_seed=100+i,
            price=price,
            print_summary=True,
            csv_filename=None
        )

        # Record results
        row = [
            i+1,
            prod_cap,
            buf_caps.tolist(),
            float(invc),
            rejoc.tolist(),
            unsatdc.tolist(),
            float(mu),
            lmd.tolist(),
            omg.tolist(),
            float(theta),
            price.tolist(),
            init_state.tolist(),
            int(best_b0),
            best_bi.tolist(),
            float(best_cost),
            float(sim_result['Total Cost']),
            float(sim_result['Total Profit']),
            sim_result['Total Orders'].tolist(),
            sim_result['Total Accepted'].tolist()
        ]
        results_rows.append(row)

    # Write summary CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in results_rows:
            writer.writerow(row)

    print(f"\nAll simulation results written to: {output_csv}")
