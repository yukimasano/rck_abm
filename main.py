from model.model_runner import runner
import argparse


INDIVIDUAL_CAPITAL = 1.02e06
COBB_DOUGLAS_B = 192.281339
INDIVIDUAL_LABOR = 1.425


def get_parser():
    parser = argparse.ArgumentParser(description="RCK ABM model")
    # network :
    # Fully-connected (FC), Watts-Strogatz (WS), Erdoes-Renyi (ER), Barabasi-Albert (BA)
    parser.add_argument(
        "--top",
        default="FC",
        choices=["FC", "WS", "ER", "BA"],
        type=str,
        help="Network topology (default: FC)",
    )
    parser.add_argument(
        "-N", "--nagents", default=500, type=int, help="number of agents (default:500)"
    )
    parser.add_argument(
        "--k",
        default=20,
        type=int,
        help="mean degree for non fully-connected graph (default: 20)",
    )
    parser.add_argument(
        "--p",
        default=0.01,
        type=float,
        help="Connection probability for WS network, ignored for other topologies (default:0.01)",
    )

    # social interaction
    parser.add_argument(
        "--tau", type=float, help="Mean interaction time tau (default: 30)"
    )

    # economic parameters
    parser.add_argument(
        "--d", type=float, help="Depreciation rate (default:0.1)"
    )

    parser.add_argument(
        "--alpha", type=float, help="Capital elasticity (default: 0.66)"
    )
    parser.add_argument(
        "--phi", type=float, help="Labor normed std. deviation (default: 0.01)"
    )
    parser.add_argument(
        "--K",
        default=INDIVIDUAL_CAPITAL,
        type=float,
        help="Initial household capital (default: %f)" % INDIVIDUAL_CAPITAL,
    )

    # additional settings
    parser.add_argument(
        "--delta-s",
        type=float,
        help="Minimum difference in s to copy from another agent (default:0)",
    )
    parser.add_argument(
        "--w-future",
        type=float,
        help="Weight of future prediction in agents decision making (default:0)",
    )

    # fixed households
    parser.add_argument(
        "--pfixed",
        type=float,
        help="Expected fraction of households with fixed savings rate (default: 0)",
    )
    parser.add_argument(
        "--rfixed",
        type=float,
        help="Discount rate in [1/yr] of households with fixed savings rate (default: 0.05)",
    )
    parser.add_argument(
        "--sfixed",
        type=float,
        help="Savings rate of fixed households (default: alpha * d / (rfixed + d))",
    )

    # exploration:
    parser.add_argument(
        "--pexplore",
        type=float,
        help="probability to explore random savings rate instead of imitating (default: 0)",
    )

    # experiment length
    parser.add_argument(
        "--tmax",
        default=20,
        type=int,
        help="Length of simulation in units of tau (default:1000)",
    )

    # house keeping
    parser.add_argument(
        "--saveloc", default="test_output/", type=str, help="where to save output"
    )
    parser.add_argument(
        "--micro",
        default=False,
        action="store_true",
        help="whether to store micro history (default: False)",
    )
    parser.add_argument(
        "--dontsave",
        default=False,
        action="store_true",
        help="whether or not to store data in separate files for each run",
    )
    parser.add_argument(
        "--logiter",
        default=100000,
        type=int,
        help="Logging every after 1M (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed (default:0 means no seed is set.)",
    )

    # movie:
    parser.add_argument(
        "--movie",
        type=str,
        help="Filename prefix for movie (default: False = no movie)",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    runner(args)
    print("..done")