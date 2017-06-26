# Plots a .tsv file. This was a simple script used to generate
# the graph of the MS-SSIM in the paper.

import numpy as np
import matplotlib.pyplot as plt


def open_tsv(file_name):
	with open(file_name, 'r') as f:
		tsv = f.readlines()

	ret = []
	for l in tsv:
		ret.append(l.split('\t'))

	print(ret)
	return ret

tsv = open_tsv('msssim._10tsv')

x = []
y = []
for i in tsv:
	x.append(i[1])
	y.append(i[4])

#N = 50
#x = np.random.rand(N)
#y = np.random.rand(N)

plt.scatter(x, y)
#plt.plot([0.28057,0], [0.28057,1])
plt.plot([0,1], [0,1])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('training data MS-SSIM value')
plt.ylabel('samples MS-SSIM value')
#plt.show()
plt.savefig('msssim.pdf', format='pdf')
