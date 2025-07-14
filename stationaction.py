import numpy as np

class StateAction:
    """StateAction class to determine the state and control action in a system."""
    def __init__(self, states, statenum, invc, rejoc, unsatdc, mu, lmd, omg, theta):
        # Initialize the system parameters
        # states is an array recording the capacity of each buffer in the system
        # statenum is the number of the system states \product_{all the buffer} (B_i)
        # current state number is determined as
        # \sum_{all the buffers} (b_i*(\product_{all the buffer in the front} B_j))
        # productioncontrol is the possible production control actions that the system can take
        # ordercontrol is the possible order arrival control
        # duecontrol is the possible order due control
        # invc is the unit inventory cost
        # rejoc is the unit rejection order cost
        # unsatdc is the unit unsatisfied order cost
        # mu is the supply rate
        # lmd is the order arrival rate
        # omg is the order due rate
        self.buffercap = states
        self.statenum = statenum
        # self.cursta = currentstates
        self.value = np.zeros(statenum)
        # self.controlpolicy = np.zeros(1 + 2 * (len(self.buffercap) - 1))
        self.ic = invc
        self.roc = rejoc
        self.udc = unsatdc
        self.mu = mu
        self.lmd = lmd
        self.omg = omg
        self.theta = theta
        self.beta = 0

    def state_to_number(self, inputstate):
        """Convert state vector to a state number."""
        sn = 0  # system state number
        prestatenumber = 1  # multiplier of buffer capacities until current index
        for i in range(len(self.buffercap)):
            sn += inputstate[i] * prestatenumber
            if i < len(self.buffercap) - 1:
                prestatenumber *= (self.buffercap[i] + 1)
        return sn

    def number_to_state(self, inputstatenumber):
        """Convert state number to a state vector."""
        remaining = inputstatenumber
        statemultiplier = 1
        systemstate = np.zeros(len(self.buffercap), dtype=int)
        for i in range(len(self.buffercap) - 1):
            statemultiplier *= (self.buffercap[i] + 1)
        for i in range(len(self.buffercap)):
            systemstate[len(self.buffercap) - i - 1] = remaining // statemultiplier
            remaining %= statemultiplier
            if i < len(self.buffercap) - 1:
                statemultiplier //= (self.buffercap[len(self.buffercap) - 2 - i] + 1)
        return systemstate

    def state_trans_value(self, inputstate):
        """Calculate the state value with the given control decisions."""
        statetrans = inputstate.copy()
        svalue = self.ic * statetrans[0]
        cd = self.control_decision(inputstate)  # control decisions based on the input state
        self.beta = self.mu + sum(self.lmd[i] + self.buffercap[i+1] * self.omg[i] for i in range(len(self.lmd)))

        for i in range(len(self.buffercap)):
            if i == 0:
                # Production control
                if inputstate[0] < cd[0]:
                    statetrans[i] = cd[0]
                statenumber = self.state_to_number(statetrans)
                pvalue = self.value[statenumber]
                svalue += self.mu * pvalue
                statetrans = inputstate.copy()
            else:
                # Order acceptance or rejection (A_i^0)
                if cd[i] == 0 and statetrans[i] < self.buffercap[i]:  # Accept the order
                    statetrans[i] += 1
                    statenumber = self.state_to_number(statetrans)
                    lvalue = self.value[statenumber]
                    #statetrans = inputstate.copy()
                else:
                    statenumber = self.state_to_number(statetrans)
                    lvalue = self.value[statenumber] + self.roc[i-1]
                    #statetrans = inputstate.copy()
                svalue += self.lmd[i-1] * lvalue
                statetrans = inputstate.copy()

                # Order satisfaction or delay (A_i^1)
                if cd[len(self.buffercap) + i - 1] == 1:  # Satisfy the order
                    if statetrans[i] > 0 and statetrans[0] > 0:
                        statetrans[i] -= 1
                        statetrans[0] -= 1
                        statenumber = self.state_to_number(statetrans)
                        ovalue1 = self.value[statenumber]
                    else:
                        statenumber = self.state_to_number(statetrans)
                        ovalue1 = self.value[statenumber] + self.udc[i-1]
                    svalue += self.omg[i-1] * inputstate[i] * ovalue1
                    statetrans = inputstate.copy()
                else:
                    statenumber = self.state_to_number(statetrans)
                    ovalue1 = self.value[statenumber] + self.udc[i - 1]
                    svalue += self.omg[i - 1] * inputstate[i] * ovalue1
                    statetrans = inputstate.copy()

                # Remaining unsatisfied orders (A_i^2)
                statenumber = self.state_to_number(inputstate)
                ovalue2 = self.value[statenumber]
                svalue += self.omg[i-1] * (self.buffercap[i] - statetrans[i]) * ovalue2

        svalue /= (self.beta + self.theta)
        return svalue

    def one_iteration(self):
        """Perform one iteration of the value iteration algorithm."""
        delta = 0
        for s in range(self.statenum):
            temp = self.value[s]
            currentstate = self.number_to_state(s)
            values = self.state_trans_value(currentstate)
            self.value[s] = values
            delta = max(delta, abs(temp - self.value[s]))
        return delta

    def control_decision(self, inputstate):
        """Determine the optimal control decisions."""
        cd = np.zeros(2 * len(self.buffercap) - 1)  # control decision output vector

        # Production control (for buffer B_0)
        best_r = inputstate[0]
        best_sn = self.state_to_number(inputstate)
        for r in range(self.buffercap[0] + 1):
            sn_r = self.state_to_number(
                np.concatenate(([r], inputstate[1:]))
            )
            if self.value[sn_r] < self.value[best_sn]:
                best_sn = sn_r
                best_r = r
        cd[0] = best_r

        # Order acceptance/rejection (A_i^0) and satisfaction/delay (A_i^1)
        for i in range(1, len(self.buffercap)):
            currentstate = inputstate.copy()
            if inputstate[i] < self.buffercap[i]:
                nextstate = currentstate.copy()
                nextstate[i] += 1
                sn0 = self.state_to_number(currentstate)
                sn = self.state_to_number(nextstate)
                if self.value[sn0] + self.roc[i-1] < self.value[sn]:
                    cd[i] = 1  # Reject the incoming order

            if currentstate[0] > 0 and currentstate[i] > 0:
                nextstate = currentstate.copy()
                nextstate[0] -= 1
                nextstate[i] -= 1
                sn0 = self.state_to_number(currentstate)
                sn = self.state_to_number(nextstate)
                if self.value[sn0] + self.udc[i-1] > self.value[sn]:
                    cd[len(self.buffercap) + i - 1] = 1  # Satisfy the order

        return cd