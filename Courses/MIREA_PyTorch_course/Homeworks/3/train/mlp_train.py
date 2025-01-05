from nn import MLP

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

n = MLP(3, [4, 4, 1])
acc = 0
for k in range(200):
    n_right_answ = 0
    b = []
    total_loss = 0
    for i in range(len(xs)):
        n.zero_grad()
        # forward
        a = n(xs[i]).data
        b.append(a)
        # calculate loss (mean square error)
        loss = (ys[i] - a)**2
        total_loss += loss
        n_right_answ += 1 if loss < 0.3 else 0
        # backward (zero_grad + backward)
        n.backward()
        # update
        learning_rate = 0.01
        for p in n.parameters():

            p.data -= learning_rate * p.grad * 2 * (a - ys[i])

    if k % 1 == 0:
        acc = round(n_right_answ / (len(xs)), 2)
        print(f"step {k} loss {total_loss}, accuracy {acc*100}%")

