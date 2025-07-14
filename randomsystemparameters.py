import numpy as np
import pandas as pd

def generate_system_params(n_products, seed=None):
    np.random.seed(seed)
    buf_caps = np.random.randint(2, 6, size=n_products)         # b_i ∈ [2,9]
    prod_cap = np.random.randint(np.max(buf_caps) + 4, np.max(buf_caps) + 6)  # Ensure prod_cap >= max buffer, similar to before
    invc = np.round(np.random.uniform(0.5, 2), 2)                # h ∈ [0.5,2]
    rejoc = np.random.randint(3, 21, size=n_products)            # c_i^r ∈ [3,20]
    unsatdc = np.random.randint(1, 11, size=n_products)          # c_i^d ∈ [1,10]
    mu = np.round(np.random.uniform(0.2, 0.6), 2)                # μ ∈ [0.2,0.6]
    lmd = np.round(np.random.uniform(0.1, 0.4, size=n_products), 3)  # λ_i ∈ [0.1,0.4]
    omg = np.round(np.random.uniform(0.1, 0.4, size=n_products), 3)  # ω_i ∈ [0.1,0.4]
    theta = np.round(np.random.uniform(0.00001, 0.0001), 3)            # Discount factor
    price = np.round(np.random.uniform(80.0, 150.0, size=n_products), 2)  # Prices
    # Initial state: [prod inventory, buffers...]
    init_state = np.concatenate([[np.random.randint(0, prod_cap + 1)],
                                 np.random.randint(0, buf_caps[0] + 1, 1),
                                 np.random.randint(0, buf_caps[1] + 1, 1) if n_products > 1 else []
                                 ])
    if n_products > 2:
        for i in range(2, n_products):
            init_state = np.concatenate([init_state, np.random.randint(0, buf_caps[i] + 1, 1)])

    params = {
        "prod_cap": prod_cap,
        "buf_caps": buf_caps.tolist(),
        "invc": invc,
        "rejoc": rejoc.tolist(),
        "unsatdc": unsatdc.tolist(),
        "mu": mu,
        "lmd": lmd.tolist(),
        "omg": omg.tolist(),
        "theta": theta,
        "price": price.tolist(),
        "init_state": init_state.tolist()
    }
    return params

def params_to_dataframe(params):
    # Flatten lists for CSV representation
    data = {k: [v if not isinstance(v, list) else str(v)] for k, v in params.items()}
    return pd.DataFrame(data)

if __name__ == "__main__":
    n_products = 4       # Use 2 products
    n_systems = 10        # Generate 2 systems

    all_df = []
    for i in range(n_systems):
        params = generate_system_params(n_products, seed=None)
        df = params_to_dataframe(params)
        df['system_id'] = i + 1
        all_df.append(df)
    all_df = pd.concat(all_df, ignore_index=True)
    all_df.to_csv("random_system_parameters.csv", index=False)
    print("Parameters saved to random_system_parameters.csv")
