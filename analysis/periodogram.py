import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
import sys


def show_periodogram(path_to_trajectory_file, **kwargs):
    trajectory = pd.read_pickle(path_to_trajectory_file)
    final = pd.read_pickle(path_to_trajectory_file.replace("traj", "final"))

    plot_periodograms(trajectory, final, path_to_trajectory_file, **kwargs)


def plot_periodograms(
    trajectory,
    final,
    path_to_trajectory_file,
    path_to_summary_output="./",
    split: int = None,
    use_astropy: bool = True,
    use_diff: bool = False,
    use_log: bool = True,
    detrend: bool = False,
    precenter: bool = True,
):

    idx = trajectory.index.values

    K = trajectory["capital"].values
    if split is None:
        # use later 40% of trajectory to compute periodicities
        split = int(len(idx) * 0.4)
    start = len(idx) - split
    end = len(idx)  # min( (len(K), 100000))

    print(f"start: {start}")
    print(f"end: {end}")
    print(f"using last {end - start} steps")
    print(f"i.e. last {idx[end - 1] - idx[start]:.3f} years")
    y = K[start:end]
    sav = ((trajectory["Y"] - trajectory["consumption"]) / trajectory["Y"]).values[start:end]

    if use_log:
        y = np.log(y)
    if use_diff:
        y = y[1:] - y[:-1]
        x = idx[start : end - 1]
    else:
        x = idx[start:end]
    if detrend:
        y = signal.detrend(y)

    taus = np.linspace(1, 200, 801)[::-1]  # 800
    omega = 2 * np.pi / taus
    f = 1 / taus

    if use_astropy:
        from astropy.timeseries import LombScargle

        ls = LombScargle(x, y, fit_mean=True, center_data=precenter)
        power = ls.power(f)
    else:
        pgram = signal.lombscargle(
            x, y, omega, normalize=True, precenter=precenter
        )  # precenter makes no big difference
        # print(sum(pgram[-100:]), flush=True)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(ncols=5, figsize=(16, 3))
    fig.suptitle("Lomb-Scargle periodogram, trajectory and savings rate distribution")
    if use_astropy:
        thresh = ls.false_alarm_level(
            0.01, minimum_frequency=f.min(), maximum_frequency=f.max()
        )
        ax2.plot(taus, power)
        ax2.plot([taus[0], taus[-1]], [thresh, thresh], "--")
        ax1.plot(taus[-100:], power[-100:])
        ax1.plot([taus[-100:][0], taus[-100:][-1]], [thresh, thresh])
    else:
        ax2.plot(taus, pgram)
        ax1.plot(taus[-100:], pgram[-100:])

    ax1.set_xlabel("Periodicity in years")
    ax1.set_ylabel("Power in periodicity")
    ax1.set_title("Periodogram 0-25y")

    ax2.set_xlabel("Periodicity in years")
    ax2.set_ylabel("Power in periodicity")
    ax2.set_title("Periodogram 0-100y")

    plot_trajectory(idx, start, end, ax3, trajectory, ax4, sav, ax5, final)

    plt.tight_layout()
    path = (
        path_to_summary_output
        + path_to_trajectory_file.split("/")[-1][1:]
        + "_summary_a.png"
    )
    print(path)
    fig.savefig(path)
    print(f"power contained in frequencies 0-25y: {sum(power[-100:]) if use_astropy else sum(pgram[-100:]):.3f}")
    plt.close()

    pgram = pgram if not use_astropy else power
    return sum(pgram[-100:]), taus, pgram


def plot_trajectory(
        idx, start, end, ax3, traj, ax4, sav, ax5, final
):
    threehundredyears = (idx[start:] - idx[start]) > 300
    if sum(threehundredyears) > 0:
        threehundredyears = np.argwhere((idx[start:] - idx[start]) > 300).T[0][0]
        end = start + threehundredyears
    ax3.plot(
        idx[start : end],
        traj["capital"].values[start : end],
        )
    ax3.set_xlabel("t in years")
    ax3.set_ylabel("K")
    ax3.set_title("K(t)")

    ax4.plot(idx[start : end], sav[:end-start])
    ax4.set_xlabel("t in years")
    ax4.set_ylabel("Economy wide savings rate")
    ax4.set_title("s(t)")
    ax5.hist(final["savings_rate"], bins=30)
    ax5.set_xlabel("savings rate")
    ax5.set_xlabel("Frequency")
    ax5.set_title("Savings rate histogram at end")



if __name__ == "__main__":
    # i.e. run this via `python periodogram.py path/to/picklefile.pkl`
    # or try it via `python periodogram.py demo`
    if sys.argv[1] == 'demo':
        import os
        os.makedirs('demo',exist_ok=True)
        if not os.path.exists('demo/_TBA_N3000_k20_d20_tau1.5_tmax1000_al66_pf5_rf0--traj.pkl'):
            print('downloading demo file (144MB)')
            url = 'https://www.dropbox.com/s/0podggqm9j1u2t7/_TBA_N3000_k20_d20_tau1.5_tmax1000_al66_pf5_rf0--traj.pkl'
            os.system(f"""wget -P demo/ '{url}'""")
            url = 'https://www.dropbox.com/s/y3k3hi6ek41dizg/_TBA_N3000_k20_d20_tau1.5_tmax1000_al66_pf5_rf0--final.pkl'
            os.system(f"""wget -P demo/  '{url}'""")
        sys.argv[1] = 'demo/_TBA_N3000_k20_d20_tau1.5_tmax1000_al66_pf5_rf0--traj.pkl'
    show_periodogram(sys.argv[1])
