import matplotlib.pyplot as plt
import numpy as np
import pickle

def get_data(filename):
	with open(filename, "rb") as fp:
		out = pickle.load(fp)
	return np.array(out)

def plot_results(diff, loss):
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 2, 1)
	ax1.plot(diff[:,0], diff[:,1])
	ax1.title("Average Difference")
	ax1.xlabel("Epochs")
	ax1.ylabel("Diff")

	ax2 = fig.add_subplot(1, 2, 2)
	ax2.plot(loss[:,0], loss[:,1])
	ax2.title("Average loss")
	ax2.xlabel("Epochs")
	ax2.ylabel("loss")
	plt.show()

if __name__ == '__main__':
	loss = get_data("loss.txt")
	diff = get_data("diff.txt")
	print(loss)
	print(diff)
	plot_results(diff, loss)