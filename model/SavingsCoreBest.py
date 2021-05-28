from scipy.integrate import odeint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import networkx as nx


class SavingsCoreBest:
    def __init__(
            self,
            adjacency=None,
            savings_rate=None,
            capital=None,
            tau=3,
            phi=0.01,
            eps=0.01,
            P=1.0,
            b=1.0,
            d=0.1,
            alpha=2.0 / 3,
            r_b=0,
            test=False,
            e_trajectory_output=True,
            macro_trajectory_output=True,
            delta_s=0.0,
            w_future=0.0,
            # for fixed households:
            pfixed=0.0,   # expected fraction of households with fixed savings rate
            rfixed=0.05,  # their discount rate, defaults to 5%/yr
                          # their savings rate, defaults to original RCK optimal
                          # savings rate given discount rate rfixed
            sfixed=None,
            # exploration:
            pexplore=0.0,
    ):

        # copying threshold
        self.delta_s = delta_s

        # future thinking parameter
        self.pfuture = w_future
        # General Parameters

        # turn output for debugging on or off
        self.debug = test
        # toggle e_trajectory output
        self.e_trajectory_output = e_trajectory_output
        self.macro_trajectory_output = macro_trajectory_output

        # movie output, default off
        self.movie = False

        # General Variables

        # System Time
        self.t = 0.0
        # Step counter for output
        self.steps = 0

        # variable to set if the model converged to some final state.
        self.converged = False
        # safes the system time at which consensus is reached
        self.convergence_time = float("NaN")
        # if not converged: opinion state at t_max
        self.convergence_state = -1

        # list to save e_trajectory of output variables
        self.e_trajectory = []
        # list to save macroscopic quantities to compare with
        # moment closure / pair based proxy approach
        self.macro_trajectory = []
        # dictionary for final state
        self.final_state = {}

        # Household parameters

        # mean waiting time between social updates
        self.tau = tau
        # the std of labor distribution, corresponds to sigma_L
        self.phi = phi
        # modulo of the maximum imitation error, corresponds to gamma
        self.eps = eps

        # number of households
        self.n = adjacency.number_of_nodes()

        # waiting times between savings rate change events for each household
        self.waiting_times = np.random.exponential(scale=self.tau, size=self.n)
        # adjacency matrix between households
        self.neighbors = nx.adj_matrix(adjacency).toarray()
        self.G = adjacency
        # investment_decisions as indices of possible_opinions
        self.savings_rate = np.array(savings_rate)

        # household capital in clean capital
        if capital is None:
            self.capital = np.ones(self.n)
        else:
            self.capital = capital

        # household income (for social update)
        self.income = np.zeros(self.n)

        # for Cobb Douglas economics:
        # Solow residual
        self.b = b
        # labor elasticity
        self.alpha = alpha
        # capital elasticity
        self.beta = 1.0 - self.alpha
        # capital depreciation rate
        self.d = d
        # population growth rate
        self.r_b = r_b

        # total capital (supply)
        self.K = self.capital.sum()
        # total labor (supply)
        # old (before 31.1.2020:) self.P = np.random.normal(float(P), (float(P)/self.n)*self.phi, self.n)
        self.P = np.random.normal(P, P * self.phi, self.n)
        self.Psum = sum(self.P)
        while any(self.P < 0):
            self.P = np.random.normal(P, P * self.phi, self.n)
        # Production
        self.Y = 0.0
        # wage
        self.w = 0.0
        # capital rent
        self.r = 0.0

        if self.e_trajectory_output:
            self.init_e_trajectory()
        if self.macro_trajectory_output:
            self.init_macro_trajectory()

        self.s_trajectory = pd.DataFrame(columns=range(self.n))

        # fixed households:
        self.pfixed = pfixed
        self.is_fixed = np.random.rand(self.n) < pfixed
        print("using", self.is_fixed.sum(), "fixed households")
        if sfixed is None:
            self.rfixed = rfixed
            sfixed = self.beta * d / (rfixed + d)
            print(f"fixed households have discount rate {rfixed:.3f}")
            print(f"and use the corresp. equil. savings rate of the orig. RCK model, {sfixed:.3f}")
        self.savings_rate[self.is_fixed] = self.sfixed = sfixed

        # exploration:
        self.pexplore = pexplore

    def c_future(self, sj, Li, Ki=0):
        """simple estimation future consumption
        * estimates how the consumption of the world looks one tau into the future
        * based on the current visible savings rates, r,w, delta and own labor Li and capital Ki"""
        r = self.r
        w = self.w
        delta = self.d
        tau = self.tau
        eta_j = r * sj - delta
        return (1.0 - sj) * (
                r * (Ki + sj * w * Li / eta_j) * np.exp(eta_j * tau)
                + w * Li * (1.0 - r * sj / eta_j)
        )

    def run(self, t_max=200.0):
        """
        run model for t<t_max or until consensus is reached

        Parameter
        ---------
        t_max : float
            The maximum time the system is integrated [Default: 100]
            before run() exits. If the model reaches consensus, or is
            unable to find further update candidated, it ends immediately

        """
        for t_max_i in tqdm(np.linspace(0, t_max, 1000)):
            while self.t < t_max_i:
                # 1 find update candidate and respective update time
                (candidate, neighbor, _, update_time) = self.find_update_candidates()

                # 2 integrate economic model until t=update_time:
                self.update_economy(update_time)

                # 3 update opinion formation in case,
                # update candidate was found:
                self.update_savings_rate(candidate, neighbor)

            # save final state to dictionary
            self.final_state = {
                "adjacency": self.neighbors,
                "savings_rate": self.savings_rate,
                "capital": self.capital,
                "tau": self.tau,
                "phi": self.phi,
                "eps": self.eps,
                "P": self.P,
                "b": self.b,
                "d": self.d,
                "test": self.debug,
            }
        return self.final_state

    def economy_dot(self, x0, t):
        """
        economic model assuming Cobb-Douglas production:

            Y = b P^pi K^kappa

        and no profits:

            Y - w P - r K = 0,

        Parameters:
        -----------

        x0  : list[float]
            state vector of the system of length
            N + 1. First N entries are
            household capital [0:n],
            the last entry is total population.
        t   : float
            the system time.

        Returns:
        --------
        x1  : list[floats]
            updated state vector of the system of length
            N + 1. First N entries are changes
            household capital [n:2N],
            the last entry is the change in total population
        """

        capital = x0[
                  : self.n
                  ]  # np.where(x0[0:self.n] > 0, x0[0:self.n], np.full(self.n, self.epsilon.eps))

        P = self.Psum
        K = capital.sum()

        self.w = self.b * self.alpha * P ** (self.alpha - 1) * K ** self.beta
        self.r = self.b * self.beta * P ** self.alpha * K ** (self.beta - 1)

        self.K = K

        self.income = self.r * self.capital + self.w * self.P

        P_dot = self.r_b * P
        capital_dot = self.savings_rate * self.income - self.capital * self.d

        return list(capital_dot) + [P_dot]

    def init_movie(self):
        self.fig = plt.figure(figsize=(6, 6), dpi=100)
        self.sss = np.linspace(0, 1, 101)[1:-1]
        (self.curve,) = plt.semilogy(self.sss, 1e2 + 0 * self.sss, "b-", alpha=0.1)
        (self.line_K,) = plt.semilogy(
            [0, 1], [self.capital.mean(), self.capital.mean()], "k-", alpha=0.5
        )
        s = np.average(self.savings_rate, weights=self.income)
        (self.line_s,) = plt.semilogy([s, s], [1e0, 1e10], "k-", alpha=0.5)
        plt.semilogy([self.sfixed, self.sfixed], [1e0, 1e10], "k--", alpha=0.5)
        (self.scattermax,) = plt.semilogy(
            [self.savings_rate[0]], [self.capital[0]], "b.", alpha=0.1, ms=30
        )
        (self.scatter,) = plt.semilogy(
            self.savings_rate[:-3], self.capital[:-3], "k.", alpha=0.1, ms=5
        )
        (self.scatterleaves,) = plt.semilogy(
            self.savings_rate[-3:], self.capital[-3:], "g.", alpha=1, ms=7
        )
        (self.scatterhub,) = plt.semilogy(
            [self.savings_rate[0]], [self.capital[0]], "r.", alpha=1, ms=10
        )
        ax = self.ax = plt.gca()
        ax.set_xlabel("savings rate")
        ax.set_ylabel("capital")
        ax.set_xlim(0, 1)
        ax.set_ylim(1e2, 1e6)
        self.figno = 0
        self.lastframet = self.t
        # explanation: consumption = (1-s) I = (1-s) (r K + w L) --> K(s|C) = (C/(1-s) - w L)/r

    def update_movie(self):
        if self.t >= self.lastframet + self.tau / 10:
            s = np.average(self.savings_rate, weights=self.income)
            self.curve.set_ydata(
                (self.consumption.max() / (1 - self.sss) - self.w * self.P[0]) / self.r
            )
            self.line_s.set_xdata([s, s])
            self.line_K.set_ydata([self.capital.mean(), self.capital.mean()])
            m = np.argmax(self.consumption)
            self.scattermax.set_xdata([self.savings_rate[m]])
            self.scattermax.set_ydata([self.capital[m]])
            self.scatter.set_xdata(self.savings_rate[1:-3])
            self.scatter.set_ydata(self.capital[1:-3])
            self.scatterleaves.set_xdata(self.savings_rate[-3:])
            self.scatterleaves.set_ydata(self.capital[-3:])
            self.scatterhub.set_xdata([self.savings_rate[0]])
            self.scatterhub.set_ydata([self.capital[0]])
            plt.title("t = %.1f" % self.t)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.savefig(self.movie + "_%05d.png" % self.figno, quality=1)
            self.figno += 1
            self.lastframet = self.t


    def update_economy(self, update_time):
        """
        Integrates the economic equations of the
        model until the system time equals the update time.

        Also keeps track of the capital return rates and estimates
        the time derivatives of capital return rates trough linear
        regression.

        Finally, appends the current system state to the system e_trajectory.

        Parameters:
        -----------
        self : object
            instance of the model class
        update_time : float
            time until which system is integrated
        """

        dt = [self.t, update_time]
        x0 = list(self.capital) + [self.Psum]

        # integrate the system
        x1 = odeint(self.economy_dot, x0, dt, mxhnil=1, mxstep=5000000)[1]

        self.capital = np.where(x1[0 : self.n] > 0, x1[0 : self.n], np.zeros(self.n))

        self.t = update_time
        self.steps += 1

        # calculate economic output:
        self.Y = self.b * self.K ** self.beta * self.Psum ** self.alpha
        self.consumption = self.income * (1 - self.savings_rate)

        # output economic data
        if self.e_trajectory_output:
            self.update_e_trajectory()
        if self.macro_trajectory_output:
            self.update_macro_trajectory()
        if self.movie:
            self.update_movie()

    def find_update_candidates(self):
        # find household with min waiting time
        candidate = np.random.randint(self.n)
        update_time = self.t + np.random.exponential(scale=self.tau / self.n)

        # load neighborhood of household i
        neighbors = list(
            self.G.neighbors(candidate)
        )  # self.neighbors[:, candidate].nonzero()[0]

        # choose best neighbor of candidate
        func_vals = (1.0 - self.savings_rate[neighbors]) * self.income[neighbors]
        # IF COPY based on highest CAPITAL:
        # func_vals = self.capital[neighbors]
        if self.pfuture != 0:
            cfut = self.c_future(
                sj=self.savings_rate[neighbors],
                Li=self.P[candidate],
                Ki=self.capital[candidate],
            )

            # debugging #########################################################
            if self.debug:
                a = self.c_future(0.1, Li=self.P[candidate], Ki=self.capital[candidate])
                b = self.c_future(0.5, Li=self.P[candidate], Ki=self.capital[candidate])
                c = self.c_future(0.9, Li=self.P[candidate], Ki=self.capital[candidate])
                abc = [np.round(10 * max(x), 2) for x in [a, b, c]]
                print(np.round(10 * max(func_vals), 2), abc)
            ###################################################################
            func_vals = func_vals + self.pfuture * cfut

        if self.is_fixed[candidate]:
            # copy from yourself:
            neighbor = candidate
        else:
            neighbor = neighbors[np.argmax(func_vals)]

        return candidate, neighbor, neighbors, update_time

    def update_savings_rate(self, candidate, neighbor):
        if self.is_fixed[candidate]:
            return 0
        if np.random.rand() < self.pexplore:
            self.savings_rate[candidate] = np.random.rand()
            return 0
        if self.fitness(neighbor) > self.fitness(candidate):
            if (
                    abs(self.savings_rate[candidate] - self.savings_rate[neighbor])
                    >= self.delta_s
            ):
                self.savings_rate[candidate] = self.savings_rate[
                                                   neighbor
                                               ] + np.random.uniform(-self.eps, self.eps)
                while (self.savings_rate[candidate] > 1) or (
                        self.savings_rate[candidate] < 0
                ):
                    # need savings_rate to stay in [0,1]
                    self.savings_rate[candidate] = (
                                                       self.savings_rate[neighbor]
                                                   ) + np.random.uniform(-self.eps, self.eps)
        return 0

    def fitness(self, agent):
        return self.income[agent] * (1 - self.savings_rate[agent])

    def init_e_trajectory(self):
        element = [
            "time",
            "w",
            "r",
            "Y",
            "indiv_savings_rate",
            "indiv_capital",
            "indiv_consumption",
        ]
        self.e_trajectory.append(element)

        self.w = self.b * self.alpha * self.Psum ** self.beta * self.K ** self.beta
        self.r = self.b * self.beta * self.Psum ** self.alpha * self.K ** self.alpha

        self.income = self.r * self.capital + self.w * self.P

        self.update_e_trajectory()

    def update_e_trajectory(self):
        element = [
            self.t,
            self.w,
            self.r,
            self.Y,
            self.savings_rate,
            self.capital,
            self.income * (1 - self.savings_rate),
            ]
        self.e_trajectory.append(element)

    def get_e_trajectory(self):
        # make up DataFrame from micro data
        columns = self.e_trajectory[0]
        trj = pd.DataFrame(self.e_trajectory[1:], columns=columns)
        trj = trj.set_index("time")

        return trj

    def init_macro_trajectory(self):
        element = ["time", "wage", "r", "capital", "consumption", "Y"]
        self.macro_trajectory.append(element)
        self.w = self.b * self.alpha * self.Psum ** self.beta * self.K ** self.beta
        self.r = self.b * self.beta * self.Psum ** self.alpha * self.K ** self.alpha

        self.income = self.r * self.capital + self.w * self.P

        self.update_macro_trajectory()

    def update_macro_trajectory(self):
        element = [
            self.t,
            self.w,
            self.r,
            self.capital.sum(),
            (self.income * (1 - self.savings_rate)).sum(),
            self.Y,
        ]
        self.macro_trajectory.append(element)

    def get_macro_trajectory(self):
        # make up DataFrame from micro data
        columns = self.macro_trajectory.pop(0)
        trj = pd.DataFrame(self.macro_trajectory, columns=columns)
        trj = trj.set_index("time")

        return trj
