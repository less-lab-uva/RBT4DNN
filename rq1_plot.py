import numpy as np  
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import json




# Sample data for 6 categories for two datasets mnist and celebahq (floating point values)
x_m = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
x_c = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
x_s = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']

with open('rbt4dnn/results/rq1_mnist.json', 'r') as file:
    mnist = json.load(file)
with open('rbt4dnn/results/rq1_celeba.json', 'r') as file:
    celeba = json.load(file)
with open('rbt4dnn/results/rq1_sgsm.json', 'r') as file:
    sgsm = json.load(file)

gen_M_fullData = mnist['gen_fulldata']

gen_M_fullreq = mnist["gen_fullreq"]
test_M = mnist['test']
total = 0
for i in test_M:
        total = total + i
print(f"avg for test_M = {total/len(test_M)}, min: {min(test_M)}, max = {max(test_M)}")
gen_M = mnist['gen']
total = 0
total_num = 0
minn = 100
maxx = 0
for i in gen_M:
        for j in i:
                total = total + j
                total_num += 1
                if j<minn:
                        minn = j
                if j>maxx:
                        maxx = j

print(f"avg for gen_M = {total/total_num}, min: {minn}, max = {maxx}")

test_C = celeba['test']
total = 0
for i in test_C:
        total = total + i
print(f"avg for test_C = {total/len(test_C)}, min: {min(test_C)}, max = {max(test_C)}")
gen_C = celeba['gen']
total = 0
total_num = 0
minn = 100
maxx = 0
for i in gen_C:
        for j in i:
                total = total + j
                total_num += 1
                if j<minn:
                        minn = j
                if j>maxx:
                        maxx = j

print(f"avg for gen_C = {total/total_num}, min: {minn}, max = {maxx}")

test_S = sgsm['test']
total = 0
for i in test_S:
        total = total + i
print(f"avg for test_S = {total/len(test_S)}, min: {min(test_S)}, max = {max(test_S)}")
gen_S = sgsm['gen']

total = 0
total_num = 0
minn = 100
maxx = 0
for i in gen_S:
        for j in i:
                total = total + j
                total_num += 1
                if j<minn:
                        minn = j
                if j>maxx:
                        maxx = j

print(f"avg for gen_S = {total/total_num}, min: {minn}, max = {maxx}")


# Bar width
bar_width = 0.25

# Define positions of bars on the x-axis
n_xm = len(x_m)
n_xc = len(x_m)
index_M = np.arange(n_xm)  # X positions for Dataset Mnist
index_C = np.arange(n_xm, 2 * n_xm)  # X positions for Dataset Celeba
index_S = np.arange(2 * n_xm, 3 * n_xm + 1) # X positions for Dataset SGSM


# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot for Dataset M
bar_test_M = ax.bar(index_M - bar_width/2, test_M, bar_width, label='Test data', color='black')
box_gen_M = ax.boxplot(gen_M, positions=index_M + bar_width/2, widths=bar_width * 0.8)
box_gen_M_fulldata = ax.boxplot(gen_M_fullData, patch_artist=True, positions=index_M + bar_width/2 + (bar_width * 0.8), widths=bar_width * 0.8)
box_gen_M_fullreq = ax.boxplot(gen_M_fullreq, patch_artist=True, positions=index_M + bar_width/2+ 2* (bar_width * 0.8), widths=bar_width * 0.8)

[box_gen_M_fulldata['boxes'][i].set_facecolor('green') for i in range(6)]
[box_gen_M_fullreq['boxes'][i].set_facecolor('pink') for i in range(6)]
# Plot for Dataset C
bars_test_C = ax.bar(index_C - bar_width/2, test_C, bar_width,  color='black')
box_gen_C = ax.boxplot(gen_C, positions=index_C + bar_width/2, widths=bar_width * 0.8)

# Plot for Dataset S
bars_test_S = ax.bar(index_S - bar_width/2, test_S, bar_width,  color='black')
box_gen_S = ax.boxplot(gen_S, positions=index_S + bar_width/2, widths=bar_width * 0.8)

# Set x-axis labels for categories and datasets
fontsize = 15
ax.set_xticks(np.concatenate([index_M, index_C, index_S]))
ax.set_xticklabels(np.concatenate([x_m, x_c, x_s]))  # Repeat the category labels for both Dataset A and Dataset B
ax.set_yticks(np.arange(0,101,step = 10))
ax.set_yticklabels(np.arange(0,101,step = 10))
ax.tick_params(axis='both', which='major', labelsize=fontsize)

ax.text(n_xm / 2 - 0.5, -15, 'MNIST', ha='center', va='center', fontsize=fontsize, color='black', fontweight='bold')
ax.text(1.5 * n_xm - 0.5, -15, 'CelebA-HQ', ha='center', va='center', fontsize=fontsize, color='black', fontweight='bold')
ax.text(2.5 * n_xm - 0.5, -15, 'SGSM', ha='center', va='center', fontsize=fontsize, color='black', fontweight='bold')

# Add a vertical dotted line to separate the two datasets
ax.axvline(n_xm - 0.3, color='black', linestyle='--')
ax.axvline(2 * n_xm - 0.5, color='black', linestyle='--')

# Set axis labels and title
ax.set_xlabel('Requirements',fontweight='bold',fontsize=fontsize)
ax.set_ylabel('Precondition Match (%)', fontweight='bold',fontsize=fontsize)

# Bold the x-ticks and y-ticks using FontProperties
for tick in ax.get_xticklabels():
    tick.set_fontsize(12)
    tick.set_fontweight('bold')  # Make x-tick labels bold

for tick in ax.get_yticklabels():
    tick.set_fontsize(12)
    tick.set_fontweight('bold')  # Make y-tick labels bold


# Adjust the bottom margin to make space for the labels
plt.subplots_adjust(bottom=0.2)

# Adjust layout to fit the plot nicely
plt.tight_layout()

# Show the plot
plt.show()

plt.savefig('Figures/rq1.png')
