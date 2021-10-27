import matplotlib.pyplot as plt
import numpy as np
import pickle

def get_data(filename):
	with open(filename, "rb") as fp:
		out = pickle.load(fp)
	return np.array(out)

def plot_results(avgloss,avgdiff,steploss,stepdiff):
	fig = plt.figure()
	ax1 = fig.add_subplot(2, 2, 1)
	ax1.plot(avgdiff[:,0], avgdiff[:,1])
	ax1.set_title("Average Difference")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Diff")

	ax2 = fig.add_subplot(2, 2, 2)
	ax2.plot(avgloss[:,0], avgloss[:,1])
	ax2.set_title("Average loss")
	ax2.set_xlabel("Epochs")
	ax2.set_ylabel("loss")

	ax3 = fig.add_subplot(2, 2, 3)
	ax3.plot(stepdiff[:,0], stepdiff[:,1])
	ax3.set_title("Stepwise Difference")
	ax3.set_xlabel("Epochs")
	ax3.set_ylabel("Diff")

	ax4 = fig.add_subplot(2, 2, 4)
	ax4.plot(steploss[:,0], steploss[:,1])
	ax4.set_title("Stepwise loss")
	ax4.set_xlabel("Epochs")
	ax4.set_ylabel("loss")
	plt.show()

if __name__ == '__main__':

	avgloss = get_data("avg_loss.txt")
	avgdiff = get_data("avg_diff.txt")

	steploss = get_data("step_loss.txt")
	stepdiff = get_data("step_diff.txt")

	plot_results(avgloss,avgdiff,steploss,stepdiff)
