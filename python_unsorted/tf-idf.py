from collections import Counter
import math


docs = [
    "I have a corgi and this corgi is amazing",
    "Not a good job, buddy",
    "Give me a hand"
]
tokens = []
for doc in docs:
    count = Counter(doc.split())
    tokens.append(count)

idf = []
for layer in tokens:
    layer_idf = []
    for token in layer:
        k = 0
        for arr in tokens:
            for x in arr:
                if x == token:
                    k += 1
                    break
        layer_idf.append(math.log(3/k + 1e-6))
    idf.append(layer_idf)
for i in range(len(tokens)):
    for j, (k, v) in enumerate(tokens[i].items()):
        print(f"{k:<10} | {v} | {idf[i][j]:.2f} | {v * idf[i][j]:.2f}")
    print()