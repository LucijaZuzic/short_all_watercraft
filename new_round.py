import numpy as np
num_convert = [6.585290808207696e-05, 8.319730598912425e-05, 9.978524482842475e-05, 0.0001146598766612077, 0.00016561568417679799, 0.00023209342060718483, 0.00027354968023082663]
pot1 = 4
pot2 = 3
nl = []
nl2 = []
for num in num_convert:
    num_new = np.round(num * (10 ** pot1), pot2)
    print(num_new, "\\times 10^{-" + str(pot1) + "}")
    print(num_new)
    nl.append(str(num_new) + " \times 10^{-" + str(pot1) + "}")
    nl2.append(num_new)
print(nl)
print(nl2)