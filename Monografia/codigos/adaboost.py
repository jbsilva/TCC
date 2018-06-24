from numpy import *

x = array([[0, 1], [1, 1], [2, 1], [3, -1], [4, -1],
       [5, -1], [6, 1], [7, 1], [8, 1], [9, -1]])
p = array([[0, 0.1], [1, 0.1], [2, 0.1], [3, 0.1], [4, 0.1],
       [5, 0.1], [6, 0.1], [7, 0.1], [8, 0.1], [9, 0.1]])
h_final = 0; Thres = zeros((4, 1)); alpha = zeros((4, 1))

for t in range(0, 3):
  err = zeros((9, 1))
  thr = array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])

  for k in range(0, 9):
    if t == 2:
      h = sign(x[:, 0] - thr[k])
    else:
      h = sign(thr[k] - x[:, 0])

    for j in range(0, 10):
      if h[j] != x[j, 1]:
        err[k] = err[k] + p[j, 1]

  for l in range(0, 9):
    if err[l] == err.min():
      indx = l
      break
    Thres[t] = thr[l]

    if t == 2:
      h = sign(x[:, 0] - thr[l])
    else:
      h = sign(thr[l] - x[:, 0])

  alpha[t] = 0.5 * log((1 - err.min()) / err.min())
  q1 = exp(-alpha[t])
  q2 = exp(alpha[t])
  Zt = 2 * sqrt(err.min() * (1 - err.min()))

  for j in range(0, 10):
    if h[j] == x[j, 1]:
      p[j, 1] = (q1 * p[j, 1]) / Zt
    else:
      p[j, 1] = (q2 * p[j, 1]) / Zt

  f = alpha[t] * (h)
  h_final = h_final + f

decision = sign(h_final)
print(decision)
