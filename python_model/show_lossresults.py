import matplotlib.pyplot as plt
import numpy as np
import pickle

def get_data(filename):
	with open(filename, "rb") as fp:
		out = pickle.load(fp)
	return out

def plot_results(diff, loss):
	fig = plt.figure()
	ax = fig.add_subplot(1, 2, 1)
	ax.plot(diff[0], diff[1])

	ax2 = fig.add_subplot(1, 2, 2)
	ax2.plot(loss[0], loss[1])
	plt.show()

if __name__ == '__main__':
	loss = get_data("loss.txt")
	diff = get_data("diff.txt")
	print(loss)
	print(diff)
	plot_results(diff, loss)