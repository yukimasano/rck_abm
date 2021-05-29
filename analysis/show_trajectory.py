"""short script to show trajectory output"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="Driver")

parser.add_argument('--path', type=str, help='path to trajectory file')
parser.add_argument('--tmin', default=2000, type=float,
                    help='min time to plot')
parser.add_argument('--tmax', default=10000, type=float,
                    help='max time to plot')

args = parser.parse_args()

data = pd.read_pickle(args.path)

fig, axes = plt.subplots(ncols=3)
fig.set_figwidth(20)
fig.set_figheight(8)

# print(data.index)
subset = (data.index > args.tmin) & (data.index < min([args.tmax, data.index.values.max()]))
data = data.iloc[subset]
data['capital'].plot(ax=axes[0])
data['consumption'].plot(ax=axes[1])
((data['Y']-data['consumption'])/data['Y']).plot(ax=axes[2])

variables = ['total capital', 'total consumption', 'economy wide savings rate']

for i, ax in enumerate(axes):
    ax.set_ylabel(variables[i])
plt.show()
