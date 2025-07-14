import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stationaction import StateAction
import csv


def run_two_product_model(
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
    # === 1) Setup & Value Iteration ===
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
    profit = 0.0
    total_orders = np.zeros(2, int)
    total_accepted = np.zeros(2, int)
    state = initial_state.copy()
    avg_lead = mu #np.exp(-1.0 / mu)

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['t'] + [f'state_{i}' for i in range(len(state))] + [f'ctrl_{i}' for i in range(5)]
        writer.writerow(header)

        for t in range(sim_time):
            ctrl = sa.control_decision(state)
            writer.writerow([t] + state.tolist() + ctrl.tolist())

            # arrivals & acceptance
            for j in range(2):
                accept_idx = 1 + j
                due_idx    = 1 + buf_caps.size + j

                if np.random.rand() < lmd[j]:
                    total_orders[j] += 1
                    if ctrl[accept_idx] == 0 and state[j+1] < states[j+1]:
                        state[j+1] += 1
                        total_accepted[j] += 1
                    else:
                        cost += rejoc[j]
                # due & satisfaction
                if state[j+1] > 0 and np.random.rand() < omg[j]*state[j+1]: #np.exp(-1.0/(omg[j]*state[j+1]+1e-20)):
                    if ctrl[due_idx] == 1 and state[0] > 0:
                        state[0] -= 1
                        state[j+1] -= 1
                        profit += price[j]
                    else:
                        cost += unsatdc[j]

            # replenishment after arrivals/due
            if np.random.rand() < avg_lead and state[0] < ctrl[0]:
                state[0] = ctrl[0]
            cost += state[0] * invc

    print("\n=== Simulation Results ===")
    print(f"Total Cost:           {cost:.2f}")
    print(f"Total Profit:         {profit:.2f}")
    print(f"Total Orders:         {total_orders}")
    print(f"Total Accepted Orders:{total_accepted}")
    print(f"Simulation logged to: {csv_filename}")
    return sa


def analyze_production_policy(sa: StateAction, prod_capacity: int, buf_caps: np.ndarray):
    # Compute switching threshold ζ0^P(b1,b2)
    b1_max, b2_max = buf_caps
    zetaP = np.zeros((b1_max+1, b2_max+1), dtype=int)
    for i in range(b1_max+1):
        for j in range(b2_max+1):
            thresh = prod_capacity
            for b0 in range(prod_capacity+1):
                ctrl_val = sa.control_decision(np.array([b0, i, j]))[0]
                # if production target equals current inventory, no new production
                if ctrl_val == b0:
                    thresh = b0
                    break
            zetaP[i, j] = thresh

    # Output full control decisions to CSV
    with open('control_decisions.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['b0', 'b1', 'b2'] + [f'ctrl_{k}' for k in range(5)] + ['threshold']
        writer.writerow(header)
        for b0 in range(prod_capacity+1):
            for b1 in range(b1_max+1):
                for b2 in range(b2_max+1):
                    ctrl_all = sa.control_decision(np.array([b0, b1, b2]))
                    thresh = zetaP[b1, b2]
                    writer.writerow([b0, b1, b2] + ctrl_all.tolist() + [thresh])

    print("Control decisions saved to control_decisions.csv")


def plot_production_threshold(zetaP: np.ndarray):
    b1_max, b2_max = zetaP.shape[0]-1, zetaP.shape[1]-1
    B1, B2 = np.meshgrid(np.arange(b1_max+1), np.arange(b2_max+1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(B1, B2, zetaP.T, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel('Backlog P1')
    ax.set_ylabel('Backlog P2')
    ax.set_zlabel('b0*')
    ax.set_title('Production Switching Surface ζ0^P')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    prod_cap = 8
    buf_caps = np.array([5, 2])
    invc = 2
    rejoc = np.array([10, 12])
    unsatdc = np.array([5, 6])
    mu = 0.6
    lmd = np.array([0.4, 0.1])
    omg = np.array([0.401, 0.101])
    theta = 0.05
    price = np.array([100.0, 100.0])
    init_state = np.array([6, 2, 1])

    sa = run_two_product_model(
        prod_cap, buf_caps, invc, rejoc, unsatdc,
        mu, lmd, omg, theta, init_state, price
    )
    analyze_production_policy(sa, prod_cap, buf_caps)
    # recompute zetaP for plotting
    b1_max, b2_max = buf_caps
    zetaP = np.zeros((b1_max+1, b2_max+1), dtype=int)
    for i in range(b1_max+1):
        for j in range(b2_max+1):
            thresh = prod_cap
            for b0 in range(prod_cap+1):
                if sa.control_decision(np.array([b0, i, j]))[0] == b0:
                    thresh = b0
                    break
            zetaP[i, j] = thresh
    plot_production_threshold(zetaP)
