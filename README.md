# Ramsey-Cass-Koopman Agent-Based Model.

As described in our paper [_Emergent inequality and endogenous dynamics in a simple behavioral macroeconomic model_](https://arxiv.org/abs/1907.02155)


## Installation
Install environment:
`conda env create -f environment.yml`

Activate environment:
`conda activate rck_abm`


## Running the model
Simply run as `python3 main.py --nagents 500 --tau 500`

Detailed usage instructions:
```
usage: main.py [-h] [--top {FC,WS,ER,BA}] [-N NAGENTS] [--k K] [--p P]
               [--tau TAU] [--tmax TMAX] [--d D] [--alpha ALPHA] [--phi PHI]
               [--K K] [--delta-s DELTA_S] [--w-future W_FUTURE]
               [--sequential] [--saveloc SAVELOC] [--micro] [--dontsave]
               [--logiter LOGITER] [--seed SEED] [--pfixed PFIXED]
               [--rfixed RFIXED] [--sfixed SFIXED] [--pexplore PEXPLORE]
               [--movie MOVIE]

RCK ABM model

optional arguments:
  -h, --help            show this help message and exit
  --top {FC,WS,ER,BA}   Network topology (default: FC)
  -N NAGENTS, --nagents NAGENTS
                        number of agents (default:500)
  --k K                 mean degree for non fully-connected graph (default:
                        20)
  --p P                 Connection probability for WS network, ignored for
                        other topologies (default:0.01)
  --tau TAU             Mean interaction time tau (default: 30)
  --tmax TMAX           Length of simulation in units of tau (default:1000)
  --d D                 Depreciation rate (default:0.1)
  --alpha ALPHA         Capital elasticity (default: 0.66)
  --phi PHI             Labor normed std. deviation (default: 0.01)
  --K K                 Initial household capital (default: 1020000.000000)
  --delta-s DELTA_S     Minimum difference in s to copy from another agent
                        (default:0)
  --w-future W_FUTURE   Weight of future prediction in agents decision making
                        (default:0)
  --saveloc SAVELOC     where to save output
  --micro               whether to store micro history (default: False)
  --dontsave            whether or not to store data in separate files for
                        each run
  --logiter LOGITER     Logging every after 1M (default: 100000)
  --seed SEED           Random seed (default:0 means no seed is set.)
  --pfixed PFIXED       Expected fraction of households with fixed savings
                        rate (default: 0)
  --rfixed RFIXED       Discount rate in [1/yr] of households with fixed
                        savings rate (default: 0)
  --sfixed SFIXED       Savings rate of fixed households (default: alpha * d /
                        (rfixed + d))
  --pexplore PEXPLORE   probability to explore random savings rate instead of
                        imitating (default: 0)
  --movie MOVIE         Filename prefix for movie (default: False = no movie)
```
  ## Evaluating it
  Show basic trajectory via: 
```
python3 analysis/show_trajectory.py --path test_output/_TFC_N500_d10_tau500.0_tmax20_al66_pf0--traj.pkl
```

  Show periodograms and trajectory
```
python3 analysis/periodogram.py test_output/_TFC_N500_d10_tau500.0_tmax20_al66_pf0--traj.pkl
```
or use the demo command provided (this will download a trajectory file of size 144MB)
```
python3 analysis/periodogram.py demo
```