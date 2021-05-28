import os
import networkx as nx
import numpy as np
import pickle as cp

from random import uniform, seed
from .SavingsCoreBest import SavingsCoreBest

INDIVIDUAL_CAPITAL = 1.02e06
COBB_DOUGLAS_B = 192.281339
INDIVIDUAL_LABOR = 1.425


def runner(args):
    if args.seed != 0:
        np.random.seed(args.seed)
        seed(args.seed)

    out_loc = args.saveloc

    ###################
    # network topology
    if args.top == "WS":
        net = nx.connected_watts_strogatz_graph(args.nagents, args.k, args.p)
        net_abbr = "_TWS_N%s_k%s_p%s" % (args.nagents, args.k, args.p)
    elif args.top == "ER":
        net = nx.erdos_renyi_graph(args.nagents, float(args.k) / args.nagents)
        net = max(nx.connected_component_subgraphs(net), key=len)
        net_abbr = "_TER_N%s_k%s" % (args.nagents, args.k)
    elif args.top == "BA":
        if args.nagents <= args.k:
            print(
                f"WARNING: resetting mean degree in BA since m<=n, (m={args.nagents}, n={args.k}) in parameter settings"
            )
            args.k = args.nagents - 1
        net = nx.barabasi_albert_graph(args.nagents, args.k)
        net_abbr = "_TBA_N%s_k%s" % (args.nagents, args.k)
    else:
        net = nx.complete_graph(args.nagents)
        net_abbr = "_TFC_N%s" % (args.nagents)

    n = len(net.nodes())
    print("Using %s agents." % n, flush=True)

    # set-up individual attributes:
    capital = np.ones(n) * args.K  # starting capital
    savings_rates = np.array([uniform(0, 1) for i in range(n)])

    input_parameters = {
        "tau": args.tau,
        "d": args.d,
        "alpha": args.alpha,
        "seq": args.sequential,
        "delta_s": args.delta_s,
        "w_future": args.w_future,
        "phi": args.phi,
        "eps": 0.01,
        "b": COBB_DOUGLAS_B,
        "P": INDIVIDUAL_LABOR,
        "pfixed": args.pfixed,
        "rfixed": args.rfixed,
        "sfixed": args.sfixed,
        "pexplore": args.pexplore,
    }
    input_parameters = {key: val for key, val in input_parameters.items() if val}

    init_conditions = (net, savings_rates, capital)

    model = SavingsCoreBest(*init_conditions, **input_parameters)
    # Turn off economic trajectory
    model.e_trajectory_output = False
    model.macro_trajectory_output = True

    # Turn on debugging
    model.debug = False

    model.movie = args.movie
    if args.movie:
        model.init_movie()

    # Run Model
    final = model.run(t_max=args.tmax * model.tau)
    trajectory = model.get_macro_trajectory()

    # saving
    save_file = os.path.join(
        out_loc,
        "%s_d%s_tau%s_tmax%s_al%s_pf%s"
        % (
            net_abbr,
            int(model.d * 100),
            args.tau,
            args.tmax,
            int(model.alpha * 100),
            int(model.pfixed * 100),
        ),
    )

    if args.micro:
        micro_trajectory = model.get_e_trajectory()

    if not args.dontsave:
        try:
            os.makedirs(out_loc)
        except:
            pass
        trajectory.to_pickle(save_file + "traj.pkl")
        with open(save_file + "final.pkl", "wb") as dumpfile:
            cp.dump(final, dumpfile)

        if args.micro:
            micro_trajectory.to_pickle(save_file + "indiv_traj.pkl")

    if args.micro:
        return final, trajectory, micro_trajectory
    return final, trajectory