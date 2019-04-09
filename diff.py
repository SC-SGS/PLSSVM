
from sklearn.datasets import load_svmlight_file
import math

data1 = load_svmlight_file("5000x2000.txt.model_double", offset = 101, n_features=5000)
data2 = load_svmlight_file("5000x2000.txt.model", offset = 101, n_features=5000)

print(data1[1][1])
klasse = 0
summe = 0
euklid = 0
for i in range(5000):
    difference = data1[1][i] - data2[1][i]
    euklid += difference * difference
    if (data1[1][i] >= 0 and data2[1][i] >= 0) or (data1[1][i] < 0 and data2[1][i] < 0):
        klasse += 1
    summe += abs(difference)

print("Euklid ", math.sqrt(euklid))
print("Summe ", summe)
print("Mittelwert", summe / 5000.0 )
print("Klasse ", klasse)