import matplotlib.pyplot as plt

with open('data.txt', 'r') as data:
    data = [tuple(line.split()) for line in data][1:]



correct_pct = list()
epochs = list()
errors = list()
for t in data:
    correct_pct.append(float(t[2]))
    epochs.appe))
 append(flscatter(errors, correct_pct)
plt.show()
