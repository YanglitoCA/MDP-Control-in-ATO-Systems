import numpy as np

class StateActionThreshold:
    """StateAction class to determine the state and control action in a system."""

    def __init__(self, states, statenum, invc, rejoc, unsatdc, mu, lmd, omg, theta):
        # Initialize system parameters
        self.buffercap = states
        self.statenum = statenum
        self.value = np.zeros(statenum)  # Value function
        self.ic = invc  # Inventory cost
        self.roc = rejoc  # Rejection order cost
        self.udc = unsatdc  # Unsatisfied order cost
        self.mu = mu  # Supply rate
        self.lmd = lmd  # Order arrival rate
        self.omg = omg  # Order due rate
        self.theta = theta  # Discount factor
        self.beta = 0  # Cached rate sum

    def state_to_number(self, inputstate):
        """Convert state vector to a state number."""
        return np.dot(
            inputstate,
            np.cumprod([1] + [cap + 1 for cap in self.buffercap[:-1]]).astype(int)
        )

    def number_to_state(self, inputstatenumber):
        """Convert state number to a state vector."""
        state = []
        for cap in reversed(self.buffercap):
            state.append(inputstatenumber % (cap + 1))
            inputstatenumber //= (cap + 1)
        return np.array(state[::-1], dtype=int)

    def state_trans_value(self, inputstate):
        """Calculate the state value for given control decisions."""
        statetrans = inputstate.copy()
        svalue = self.ic * statetrans[0]
        cd = self.control_decision(inputstate)  # Control decisions
        self.beta = self.mu + np.sum(self.lmd + self.buffercap[1:] * self.omg)

        for i, cap in enumerate(self.buffercap):
            if i == 0:  # Production control
                statetrans[i] = cd[0]
                statenumber = self.state_to_number(statetrans)
                svalue += self.mu * self.value[statenumber]
                statetrans[i] = inputstate[i]
            else:
                # Order acceptance/rejection (A_i^0)
                if cd[i] == 0 and statetrans[i] < cap:  # Accept order
                    statetrans[i] += 1
                statenumber = self.state_to_number(statetrans)
                lvalue = self.value[statenumber] + (self.roc[i - 1] if cd[i] else 0)
                svalue += self.lmd[i - 1] * lvalue
                statetrans[i] = inputstate[i]

                # Order satisfaction/delay (A_i^1)
                if cd[len(self.buffercap) + i - 1] == 1:  # Satisfy order
                    if statetrans[i] > 0 and statetrans[0] > 0:
                        statetrans[i] -= 1
                        statetrans[0] -= 1
                    statenumber = self.state_to_number(statetrans)
                    ovalue1 = self.value[statenumber]
                    svalue += self.omg[i - 1] * inputstate[i] * ovalue1
                    statetrans[i] = inputstate[i]
                    statetrans[0] = inputstate[0]
                else:
                    statenumber = self.state_to_number(statetrans)
                    ovalue1 = self.value[statenumber] + self.udc[i - 1]
                    svalue += self.omg[i - 1] * inputstate[i] * ovalue1

                # Remaining unsatisfied orders (A_i^2)
                ovalue2 = self.value[statenumber]
                svalue += self.omg[i - 1] * (cap - statetrans[i]) * ovalue2

        return svalue / (self.beta + self.theta)

    def one_iteration(self):
        """Perform one iteration of the value iteration algorithm."""
        delta = 0
        for s in range(self.statenum):
            temp = self.value[s]
            currentstate = self.number_to_state(s)
            self.value[s] = self.state_trans_value(currentstate)
            delta = max(delta, abs(temp - self.value[s]))
        return delta

    def control_decision(self, inputstate):
        """Determine the optimal control decisions."""
        cd = np.zeros(2 * len(self.buffercap) - 1)  # Control decision vector

        # Production control
        next_states = np.arange(self.buffercap[0] + 1)
        next_values = [
            self.value[self.state_to_number(np.concatenate(([p], inputstate[1:])))]
            for p in next_states
        ]
        cd[0] = next_states[np.argmin(next_values)]

        # Order control
        for i in range(1, len(self.buffercap)):
            if inputstate[i] < self.buffercap[i]:  # Reject order if beneficial
                reject_value = (
                    self.value[self.state_to_number(inputstate)] + self.roc[i - 1]
                )
                accept_state = inputstate.copy()
                accept_state[i] += 1
                accept_value = self.value[self.state_to_number(accept_state)]
                if reject_value > accept_value:
                    cd[i] = 1  # Reject

            # Satisfy orders
            if inputstate[0] > 0 and inputstate[i] > 0:
                satisfy_state = inputstate.copy()
                satisfy_state[0] -= 1
                satisfy_state[i] -= 1
                satisfy_value = self.value[self.state_to_number(satisfy_state)]
                current_value = self.value[self.state_to_number(inputstate)]
                if satisfy_value > current_value:
                    cd[len(self.buffercap) + i - 1] = 1  # Satisfy order

        return cd