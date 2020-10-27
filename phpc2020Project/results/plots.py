import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)

df = pd.read_csv("strong_scaling.csv", header=None, names=['Implementation', 'Matrix Dimension', 'Processes/Threads', 'Product Time [s]', 'Total Time [s]'])
df.Implementation = df.Implementation.map(lambda x: {'seq': 'Sequential (Baseline)', 'mpi': 'MPI', 'cuda': 'CUDA'}[x])
baseline = df[df['Implementation'] == 'Sequential (Baseline)'].drop(['Processes/Threads'], axis=1).assign(key=1).merge(df[['Processes/Threads']].drop(0).assign(key=1), on="key").drop("key", axis=1)
df = df.drop(0)


fig, ax = plt.subplots(figsize=(8,10))
ax = sns.lineplot(hue='Implementation', x='Processes/Threads', y='Product Time [s]', style='Implementation', dashes=[[4, 2]], data=baseline, linewidth=5.0, markersize=12, palette=['#CC4545'], ax=ax)
ax = sns.lineplot(hue='Implementation', x='Processes/Threads', y='Product Time [s]', markers=True, style='Implementation', dashes=False, data=df, linewidth=5.0, markersize=12, palette=['#2A617D', '#9FCC45'], ax=ax)
ax.loglog()
handles, labels = ax.get_legend_handles_labels()
handles=[handles[1]]+handles[3:]
labels=[labels[1]]+labels[3:]
for h in handles:
	h.set_linewidth(5)
	h.set_markersize(12)
ax.legend(handles=handles, labels=labels,loc='upper right', ncol=1, bbox_to_anchor=(1.09, 1.09))
plt.ylim(1e-5, 1e-2)
sns.despine(left=True, bottom=True)
fig.savefig('strong_scaling_product.pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(8,10))
ax = sns.lineplot(hue='Implementation', x='Processes/Threads', y='Total Time [s]', style='Implementation', dashes=[[4, 2]], data=baseline, linewidth=5.0, markersize=12, palette=['#CC4545'], ax=ax)
ax = sns.lineplot(hue='Implementation', x='Processes/Threads', y='Total Time [s]', markers=True, style='Implementation', dashes=False, data=df, linewidth=5.0, markersize=12, palette=['#2A617D', '#9FCC45'], ax=ax)
ax.loglog()
handles, labels = ax.get_legend_handles_labels()
handles=[handles[1]]+handles[3:]
labels=[labels[1]]+labels[3:]
for h in handles:
	h.set_linewidth(5)
	h.set_markersize(12)
ax.legend(handles=handles, labels=labels,loc='upper right', ncol=1, bbox_to_anchor=(1.09, 1.09))
plt.ylim(6, 1e3)
sns.despine(left=True, bottom=True)
fig.savefig('strong_scaling_total.pdf', bbox_inches='tight')


df = pd.read_csv("weak_scaling.csv", header=None, names=['Implementation', 'Matrix Dimension', 'Processes/Threads', 'Product Time [s]', 'Total Time [s]'])
df.Implementation = df.Implementation.map(lambda x: {'seq': 'Sequential (Baseline)', 'mpi': 'MPI', 'cuda': 'CUDA'}[x])

fig, ax = plt.subplots(figsize=(8,10))
ax = sns.lineplot(hue='Implementation', x='Matrix Dimension', y='Product Time [s]', style='Implementation', dashes=[[4, 2]], data=df[df['Implementation'] == 'Sequential (Baseline)'], linewidth=5.0, markersize=12, palette=['#CC4545'], ax=ax)
ax = sns.lineplot(hue='Implementation', x='Matrix Dimension', y='Product Time [s]', markers=True, style='Implementation', dashes=False, data=df[df['Implementation'] != 'Sequential (Baseline)'], linewidth=5.0, markersize=12, palette=['#2A617D', '#9FCC45'], ax=ax)
ax.loglog()
handles, labels = ax.get_legend_handles_labels()
handles=[handles[1]]+handles[3:]
labels=[labels[1]]+labels[3:]
for h in handles:
	h.set_linewidth(5)
	h.set_markersize(12)
ax.legend(handles=handles, labels=labels,loc='upper right', ncol=1, bbox_to_anchor=(1.09, 1.09))
plt.xlim(1e1, 1.5*1e4)
plt.ylim(.8*1e-6, 1e-3)
sns.despine(left=True, bottom=True)
fig.savefig('weak_scaling_product.pdf', bbox_inches='tight')


fig, ax = plt.subplots(figsize=(8,10))
ax = sns.lineplot(hue='Implementation', x='Matrix Dimension', y='Total Time [s]', style='Implementation', dashes=[[4, 2]], data=df[df['Implementation'] == 'Sequential (Baseline)'], linewidth=5.0, markersize=12, palette=['#CC4545'], ax=ax)
ax = sns.lineplot(hue='Implementation', x='Matrix Dimension', y='Total Time [s]', markers=True, style='Implementation', dashes=False, data=df[df['Implementation'] != 'Sequential (Baseline)'], linewidth=5.0, markersize=12, palette=['#2A617D', '#9FCC45'], ax=ax)
ax.loglog()
handles, labels = ax.get_legend_handles_labels()
handles=[handles[1]]+handles[3:]
labels=[labels[1]]+labels[3:]
for h in handles:
	h.set_linewidth(5)
	h.set_markersize(12)
ax.legend(handles=handles, labels=labels,loc='upper right', ncol=1, bbox_to_anchor=(1.09, 1.09))
plt.xlim(1e1, 1.5*1e4)
plt.ylim(.4*1e-4, 1e3)
sns.despine(left=True, bottom=True)
fig.savefig('weak_scaling_total.pdf', bbox_inches='tight')

