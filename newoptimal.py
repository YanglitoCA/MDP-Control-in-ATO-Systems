import numpy as np
import pandas as pd
import ast
import csv
from stationaction import StateAction

def parse_list_column(s):
    """Helper to parse string representation of list in csv, e.g. '[1,2,3]'."""
    return np.array(ast.literal_eval(s))

def run_n_product_model(
        prod_capacity: int,
        buf_caps: np.ndarray,
        invc: float,
        rejoc: np.ndarray,
        unsatdc: np.ndarray,
        mu: float,
        lmd: np.ndarray,
        omg: np.ndarray,
        theta: float,
        initial_state: np.ndarray,
        price: np.ndarray,
        sim_time: int = 100_000,
        csv_filename: str = 'simulation_log.csv'
):
    n = len(buf_caps)  # number of products
    states = np.concatenate(([prod_capacity], buf_caps))
    statenum = int(np.prod(states + 1))
    sa = StateAction(states, statenum, invc, rejoc, unsatdc, mu, lmd, omg, theta)

    print("Running value iteration…")
    for it in range(1, 10_001):
        δ = sa.one_iteration()
        if δ < 1e-6:
            print(f" → Converged in {it} iters, δ={δ:.2e}")
            break

    # === 2) Simulation & CSV Logging ===
    cost = 0.0
    discounted_cost = 0.0
    profit = 0.0
    discounted_profit = 0.0
    total_orders = np.zeros(n, int)
    total_accepted = np.zeros(n, int)
    state = initial_state.copy()
    avg_lead = mu
    gamma = 1 / (1 + theta)  # Discount factor

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['t'] + [f'state_{i}' for i in range(len(state))] + [f'ctrl_{i}' for i in range(1 + 2 * n)]
        writer.writerow(header)

        for t in range(sim_time):
            discount = gamma ** t

            ctrl = sa.control_decision(state)
            writer.writerow([t] + state.tolist() + ctrl.tolist())

            # arrivals & acceptance
            for j in range(n):
                accept_idx = 1 + j
                due_idx = 1 + n + j

                if np.random.rand() < lmd[j]:
                    total_orders[j] += 1
                    if ctrl[accept_idx] == 0 and state[j+1] < states[j+1]:
                        state[j+1] += 1
                        total_accepted[j] += 1
                    else:
                        cost += rejoc[j]
                        discounted_cost += discount * rejoc[j]
                # due & satisfaction
                if state[j+1] > 0 and np.random.rand() < omg[j]*state[j+1]:
                    if ctrl[due_idx] == 1 and state[0] > 0:
                        state[0] -= 1
                        state[j+1] -= 1
                        profit += price[j]
                        discounted_profit += discount * price[j]
                    else:
                        cost += unsatdc[j]
                        discounted_cost += discount * unsatdc[j]

            # replenishment after arrivals/due
            if np.random.rand() < avg_lead and state[0] < ctrl[0]:
                state[0] = ctrl[0]
            cost += state[0] * invc
            discounted_cost += discount * state[0] * invc

    avg_discounted_cost = discounted_cost / sim_time
    avg_discounted_profit = discounted_profit / sim_time

    print("\n=== Simulation Results ===")
    print(f"Total Cost:                     {cost:.2f}")
    print(f"Total Profit:                   {profit:.2f}")
    print(f"Total Orders:                   {total_orders}")
    print(f"Total Accepted Orders:          {total_accepted}")
    print(f"Discounted Total Cost:          {discounted_cost:.2f}")
    print(f"Discounted Total Profit:        {discounted_profit:.2f}")
    print(f"Discounted Avg Cost per time:   {avg_discounted_cost:.6f}")
    print(f"Discounted Avg Profit per time: {avg_discounted_profit:.6f}")
    print(f"Simulation logged to:           {csv_filename}")
    return sa

def run_from_csv(csv_file, sim_time=100_000):
    df = pd.read_csv(csv_file)
    for idx, row in df.iterrows():
        print(f"\n=== Running simulation for system_id {row.get('system_id', idx+1)} ===")
        # Parse all list-type columns
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

        csv_filename = f"sim_log_system_{row.get('system_id', idx+1)}.csv"
        run_n_product_model(
            prod_cap, buf_caps, invc, rejoc, unsatdc,
            mu, lmd, omg, theta, init_state, price,
            sim_time=5000,
            csv_filename=csv_filename
        )
        print(f"Simulation for system {row.get('system_id', idx+1)} finished.\n")

if __name__ == "__main__":
    csv_file = "random_system_parameters.csv"   # Make sure this matches your parameters CSV name!
    run_from_csv(csv_file)