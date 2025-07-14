import numpy as np
import pandas as pd
import ast
import csv
from itertools import product

def parse_list_column(s):
    return np.array(ast.literal_eval(s))

class ThresholdHeuristicMDP:
    def __init__(self, prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta,
                 b0_star=None, bi_star=None, price=None):
        self.prod_cap = prod_cap
        self.buf_caps = np.array(buf_caps)
        self.buffercap = np.concatenate(([prod_cap], self.buf_caps))
        self.invc = invc
        self.rejoc = rejoc
        self.unsatdc = unsatdc
        self.mu = mu
        self.lmd = lmd
        self.omg = omg
        self.theta = theta
        self.n = len(self.buffercap)
        self.state_size = int(np.prod(self.buffercap + 1))
        self.b0_star = prod_cap if b0_star is None else b0_star
        self.bi_star = self.buf_caps if bi_star is None else np.array(bi_star)
        self.beta = self.mu + sum(self.lmd[i] + self.buffercap[i+1] * self.omg[i] for i in range(len(self.lmd)))
        self.V = np.zeros(self.state_size)
        if price is not None:
            self.price = np.array(price)
        else:
            self.price = np.zeros(self.n - 1)

    def control_decision(self, state, b0_star=None, bi_star=None):
        if b0_star is None:
            b0_star = self.b0_star
        if bi_star is None:
            bi_star = self.bi_star
        control = []
        # Production
        control.append(1 if state[0] < b0_star else 0)
        # Order acceptance
        for i in range(1, self.n):
            control.append(1 if state[i] < bi_star[i-1] else 0)
        # Fulfillment
        for i in range(1, self.n):
            control.append(1 if state[0] > 0 and state[i] > 0 else 0)
        return control

def simulate_system_threshold(
    solver, init_state, price, sim_time=100_000, random_seed=0,
    b0_star=None, bi_star=None,
    print_summary=True, csv_filename=None
):
    np.random.seed(random_seed)
    n = solver.n - 1
    state = np.array(init_state, dtype=int)
    total_cost = 0.0
    total_profit = 0.0
    total_orders = np.zeros(n, int)
    total_accepted = np.zeros(n, int)
    orders_rejected = np.zeros(n, int)
    unsatisfied = np.zeros(n, int)
    invc = solver.invc
    rejoc = solver.rejoc
    unsatdc = solver.unsatdc
    lmd = solver.lmd
    omg = solver.omg
    mu = solver.mu
    theta = solver.theta
    if price is None:
        price = np.zeros(n)
    avg_lead = mu

    gamma = 1 / (1 + theta)  # Discount factor

    if csv_filename:
        csvfile = open(csv_filename, 'w', newline='')
        writer = csv.writer(csvfile)
        header = ['t'] + [f'state_{i}' for i in range(len(state))] + [f'ctrl_{i}' for i in range(1+2*n)] \
                 + [f'accepted_{j+1}' for j in range(n)] \
                 + [f'rejected_{j+1}' for j in range(n)] \
                 + [f'fulfilled_{j+1}' for j in range(n)] \
                 + [f'unsatisfied_{j+1}' for j in range(n)]
        writer.writerow(header)
    else:
        writer = None

    for t in range(sim_time):
        discount = gamma ** t  # Discount for this step

        ctrl = solver.control_decision(state, b0_star, bi_star)
        accepted = np.zeros(n, int)
        rejected = np.zeros(n, int)
        fulfilled = np.zeros(n, int)
        unsat = np.zeros(n, int)

        # Arrivals and acceptance
        for j in range(n):
            accept_idx = 1 + j
            due_idx    = 1 + n + j
            if np.random.rand() < lmd[j]:
                total_orders[j] += 1
                if ctrl[accept_idx] == 1 and state[j+1] < solver.buffercap[j+1]:
                    state[j+1] += 1
                    total_accepted[j] += 1
                    accepted[j] = 1
                else:
                    total_cost += discount * rejoc[j]
                    orders_rejected[j] += 1
                    rejected[j] = 1
            # Due
            if state[j+1] > 0 and np.random.rand() < omg[j]*state[j+1]:
                if ctrl[due_idx] == 1 and state[0] > 0:
                    state[0] -= 1
                    state[j+1] -= 1
                    total_profit += discount * price[j]
                    fulfilled[j] = 1
                else:
                    total_cost += discount * unsatdc[j]
                    unsatisfied[j] += 1
                    unsat[j] = 1
        # Production (replenish if below threshold)
        if ctrl[0] == 1 and np.random.rand() < avg_lead and state[0] < (b0_star if b0_star is not None else solver.b0_star):
            state[0] = (b0_star if b0_star is not None else solver.b0_star)
        total_cost += discount * state[0] * invc
        # Boundaries
        for k in range(len(state)):
            state[k] = max(0, min(state[k], solver.buffercap[k]))

        if writer:
            writer.writerow([t] + state.tolist() + list(ctrl)
                            + accepted.tolist()
                            + rejected.tolist()
                            + fulfilled.tolist()
                            + unsat.tolist())

    if writer:
        csvfile.close()

    if print_summary:
        print("\n=== Simulation Results (Discounted) ===")
        print(f"Discounted Total Cost:   {total_cost:.2f}")
        print(f"Discounted Total Profit: {total_profit:.2f}")
        print(f"Total Orders:            {total_orders}")
        print(f"Total Accepted Orders:   {total_accepted}")
        print(f"Total Rejected Orders:   {orders_rejected}")
        print(f"Total Unsatisfied:       {unsatisfied}")
        print(f"Average cost per time unit: {total_cost / sim_time:.4f}")
    return {
        "Discounted Total Cost": total_cost,
        "Discounted Total Profit": total_profit,
        "Total Orders": total_orders,
        "Total Accepted Orders": total_accepted,
        "Total Rejected Orders": orders_rejected,
        "Total Unsatisfied": unsatisfied,
        "Average Cost Per Time Unit": total_cost / sim_time
    }

def optimize_thresholds_grid_search(
    prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta, price,
    sim_time=2000, verbose=False
):
    n = len(buf_caps)
    best_cost = float('inf')
    best_b0_star = None
    best_bi_star = None

    solver = ThresholdHeuristicMDP(prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta, price=price)
    b0_candidates = range(prod_cap+1)
    bi_candidates = [range(b+1) for b in buf_caps]

    for b0_star in b0_candidates:
        for bi_star_tuple in product(*bi_candidates):
            cost_res = simulate_system_threshold(
                solver, init_state=np.zeros(n+1, dtype=int), price=price,
                sim_time=sim_time, random_seed=123, b0_star=b0_star, bi_star=np.array(bi_star_tuple),
                print_summary=False, csv_filename=None)
            avg_cost = cost_res["Average Cost Per Time Unit"]
            if avg_cost < best_cost:
                best_cost = avg_cost
                best_b0_star = b0_star
                best_bi_star = np.array(bi_star_tuple)
    return best_b0_star, best_bi_star, best_cost

if __name__ == "__main__":
    csv_file = "random_system_parameters.csv"
    df = pd.read_csv(csv_file)
    system_summaries = []
    for idx, row in df.iterrows():
        print(f"\n=== Running system_id {row.get('system_id', idx+1)} ===")
        prod_cap = int(row['prod_cap'])
        buf_caps = parse_list_column(row['buf_caps'])
        invc = float(row['invc'])
        rejoc = parse_list_column(row['rejoc'])
        unsatdc = parse_list_column(row['unsatdc'])
        mu = float(row['mu'])
        lmd = parse_list_column(row['lmd'])
        omg = parse_list_column(row['omg'])
        theta = float(row['theta'])
        price = parse_list_column(row['price'])
        init_state = parse_list_column(row['init_state'])

        best_b0, best_bi, best_cost = optimize_thresholds_grid_search(
            prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta, price,
            sim_time=2000, verbose=False
        )
        print(f"\nBest thresholds found: b0*={best_b0}, bi*={best_bi}, avg_cost={best_cost:.4f}")

        solver = ThresholdHeuristicMDP(prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, theta,
                                       b0_star=best_b0, bi_star=best_bi, price=price)
        summary = simulate_system_threshold(
            solver,
            init_state=init_state,
            price=price,
            sim_time=5000,
            random_seed=idx+1,
            b0_star=best_b0,
            bi_star=best_bi,
            print_summary=True,
            csv_filename=f"sim_log_system_{row.get('system_id', idx+1)}.csv"
        )
        summary['system_id'] = row.get('system_id', idx+1)
        summary['b0_star'] = best_b0
        summary['bi_star'] = list(best_bi)
        system_summaries.append(summary)
    print("\n==== Summary Table ====")
    summary_df = pd.DataFrame(system_summaries)
    print(summary_df)
