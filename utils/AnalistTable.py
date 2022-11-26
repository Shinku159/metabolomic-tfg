import string
import pandas as pd
import numpy as np
import numpy.core.defchararray as np_f
import tensorboard as tb

experiment_id = "d4PuMINcQ3OUJz5mubc8Rw"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
df = df.sort_values(by='value', ascending=False)
val = df[df.run.str.contains('validation')]
val = val[val.step == 200]
valF1 = val[val.tag == 'evaluation_custom_f1_vs_iterations']
valACC = val[val.tag == 'evaluation_accuracy_vs_iterations']
valF1 = valF1.head(10)
valACC = valACC.head(10)

trainACC = df[df.run.isin([s.replace('validation' , 'train') for s in valACC.run.values])]
trainF1 = df[df.run.isin([s.replace('validation' , 'train') for s in valF1.run.values])]
trainACC = trainACC[trainACC.step == 199]
trainF1 = trainF1[trainF1.step == 199]
trainACC = trainACC[trainACC.tag == 'epoch_accuracy']
trainF1 = trainF1[trainF1.tag == 'epoch_custom_f1']

# valnames = np.concatenate(val.value.values, axis=0).astype(string)

print("VALIDATION ACC:")
for n in valACC.run.values:
    n = n.replace("gcms-conv-", "")
    n = n.replace("1e-05-", "")
    n = n.replace("\\validation", "")
    print(n)
for n in valACC.value.values:
    print("{:.4f}".format(n).replace(".", ","))


print("VALIDATION F1:")
for n in valF1.run.values:
    n = n.replace("gcms-conv-", "")
    n = n.replace("1e-05-", "")
    n = n.replace("\\validation", "")
    print(n)
for n in valF1.value.values:
    print("{:.4f}".format(n).replace(".", ","))

print("TRAIN ACC:")
for n in trainACC.run.values:
    n = n.replace("gcms-conv-", "")
    n = n.replace("1e-05-", "")
    n = n.replace("\\train", "")
    print(n)
for n in trainACC.value.values:
    print("{:.4f}".format(n).replace(".", ","))

print("TRAIN F1:")
for n in trainF1.run.values:
    n = n.replace("gcms-conv-", "")
    n = n.replace("1e-05-", "")
    n = n.replace("\\train", "")
    print(n)
for n in trainF1.value.values:
    print("{:.4f}".format(n).replace(".", ","))