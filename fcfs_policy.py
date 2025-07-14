import numpy as np
import pandas as pd
import ast
import csv

def parse_list_column(s):
    return np.array(ast.literal_eval(s))

def simulate_system_fcfs(
    prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, price,
    init_state, theta, sim_time=100_000, random_seed=0,
    print_summary=True, csv_filename=None
):
    np.random.seed(random_seed)
    n = len(buf_caps)
    state = np.array(init_state, dtype=int)
    total_cost = 0.0
    total_profit = 0.0
    total_orders = np.zeros(n, int)
    total_accepted = np.zeros(n, int)
    orders_rejected = np.zeros(n, int)
    unsatisfied = np.zeros(n, int)
    if price is None:
        price = np.zeros(n)
    avg_lead = mu

    gamma = 1 / (1 + theta)  # Discrete-time discount factor

    if csv_filename:
        csvfile = open(csv_filename, 'w', newline='')
        writer = csv.writer(csvfile)
        header = ['t'] + [f'state_{i}' for i in range(len(state))] \
                 + [f'accepted_{j+1}' for j in range(n)] \
                 + [f'rejected_{j+1}' for j in range(n)] \
                 + [f'fulfilled_{j+1}' for j in range(n)] \
                 + [f'unsatisfied_{j+1}' for j in range(n)]
        writer.writerow(header)
    else:
        writer = None

    for t in range(sim_time):
        accepted = np.zeros(n, int)
        rejected = np.zeros(n, int)
        fulfilled = np.zeros(n, int)
        unsat = np.zeros(n, int)
        discount = gamma**t

        # --- FCFS production: refill prod buffer if below prod_cap ---
        if state[0] < prod_cap and np.random.rand() < avg_lead:
            state[0] = prod_cap

        # --- Arrivals and acceptance ---
        for j in range(n):
            if np.random.rand() < lmd[j]:
                total_orders[j] += 1
                if state[j+1] < buf_caps[j]:
                    state[j+1] += 1
                    total_accepted[j] += 1
                    accepted[j] = 1
                else:
                    total_cost += discount * rejoc[j]
                    orders_rejected[j] += 1
                    rejected[j] = 1

        # --- Fulfillment (order due and possible satisfaction) ---
        for j in range(n):
            if state[j+1] > 0 and np.random.rand() < omg[j] * state[j+1]:
                if state[0] > 0:
                    state[0] -= 1
                    state[j+1] -= 1
                    total_profit += discount * price[j]
                    fulfilled[j] = 1
                else:
                    total_cost += discount * unsatdc[j]
                    unsatisfied[j] += 1
                    unsat[j] = 1

        # --- Holding cost for production buffer ---
        total_cost += discount * state[0] * invc

        # --- Boundaries ---
        state[0] = max(0, min(state[0], prod_cap))
        for k in range(n):
            state[k+1] = max(0, min(state[k+1], buf_caps[k]))

        if writer:
            writer.writerow(
                [t] + state.tolist()
                + accepted.tolist()
                + rejected.tolist()
                + fulfilled.tolist()
                + unsat.tolist()
            )

    if writer:
        csvfile.close()

    # Additional average discounted measures
    avg_discounted_cost = total_cost / sim_time
    avg_discounted_profit = total_profit / sim_time

    if print_summary:
        print("\n=== FCFS Simulation Results (Discounted) ===")
        print(f"Discounted Total Cost:        {total_cost:.2f}")
        print(f"Discounted Total Profit:      {total_profit:.2f}")
        print(f"Discounted Avg Cost/Time:     {avg_discounted_cost:.6f}")
        print(f"Discounted Avg Profit/Time:   {avg_discounted_profit:.6f}")
        print(f"Total Orders:                 {total_orders}")
        print(f"Total Accepted Orders:        {total_accepted}")
        print(f"Total Rejected Orders:        {orders_rejected}")
        print(f"Total Unsatisfied:            {unsatisfied}")

    return {
        "Discounted Total Cost": total_cost,
        "Discounted Total Profit": total_profit,
        "Discounted Avg Cost/Time": avg_discounted_cost,
        "Discounted Avg Profit/Time": avg_discounted_profit,
        "Total Orders": total_orders,
        "Total Accepted Orders": total_accepted,
        "Total Rejected Orders": orders_rejected,
        "Total Unsatisfied": unsatisfied,
    }

if __name__ == "__main__":
    csv_file = "random_system_parameters.csv"
    df = pd.read_csv(csv_file)
    system_summaries = []
    for idx, row in df.iterrows():
        print(f"\n=== Running system_id {row.get('system_id', idx+1)} (FCFS, discounted) ===")
        prod_cap = int(row['prod_cap'])
        buf_caps = parse_list_column(row['buf_caps'])
        invc = float(row['invc'])
        rejoc = parse_list_column(row['rejoc'])
        unsatdc = parse_list_column(row['unsatdc'])
        mu = float(row['mu'])
        lmd = parse_list_column(row['lmd'])
        omg = parse_list_column(row['omg'])
        price = parse_list_column(row['price'])
        theta = float(row['theta'])
        init_state = parse_list_column(row['init_state'])

        summary = simulate_system_fcfs(
            prod_cap, buf_caps, invc, rejoc, unsatdc, mu, lmd, omg, price,
            init_state=init_state,
            theta=theta,
            sim_time=5000,
            random_seed=idx+1,
            print_summary=True,
            csv_filename=f"sim_log_fcfs_system_{row.get('system_id', idx+1)}.csv"
        )
        summary['system_id'] = row.get('system_id', idx+1)
        system_summaries.append(summary)
    print("\n==== FCFS Summary Table (Discounted) ====")
    summary_df = pd.DataFrame(system_summaries)
    print(summary_df)
