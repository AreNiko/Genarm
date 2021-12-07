import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

def get_data(filename):
	with open(filename, "rb") as fp:
		out = pickle.load(fp)
	return np.array(out)

def multiple_plot_results(avgloss,avgdiff,steploss,stepdiff):
	fig = plt.figure()
	ax1 = fig.add_subplot(2, 2, 1)
	ax1.plot(avgdiff[:,0], avgdiff[:,1])
	ax1.set_title("Average Difference")
	ax1.set_xlabel("Epochs")
	ax1.set_ylabel("Diff")
	ax1.set_yscale('log')

	ax2 = fig.add_subplot(2, 2, 2)
	ax2.plot(avgloss[:,0], avgloss[:,1])
	ax2.set_title("Average loss")
	ax2.set_xlabel("Epochs")
	ax2.set_ylabel("loss")
	ax2.set_yscale('log')

	ax3 = fig.add_subplot(2, 2, 3)
	ax3.plot(stepdiff[:,0], stepdiff[:,1])
	ax3.set_title("Stepwise Difference")
	ax3.set_xlabel("Epochs")
	ax3.set_ylabel("Diff")
	ax3.set_yscale('log')

	ax4 = fig.add_subplot(2, 2, 4)
	ax4.plot(steploss[:,0], steploss[:,1])
	ax4.set_title("Stepwise loss")
	ax4.set_xlabel("Epochs")
	ax4.set_ylabel("loss")
	ax4.set_yscale('log')


def plot_results(data, title, stepwise=False, loss=False):
	plt.figure()
	plt.plot(data[:,0], data[:,1])
	plt.title(title)
	if stepwise:
		plt.xlabel("Steps")
	else:
		plt.xlabel("Epochs")
	if loss:
		plt.ylabel("Loss")
	else:
		plt.ylabel("Diff")
	plt.yscale('log')

def parse_args():
	"""Parse command line argument."""
	parser = argparse.ArgumentParser("Train segmention model on 3D structures.")
	parser.add_argument("test_number", help="logs the result files to specific runs")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	#avgloss = get_data("results/" + args.test_number + "/avg_train_loss.txt")
	#avgdiff = get_data("results/" + args.test_number + "/avg_train_diff.txt")

	steploss = get_data("results/" + args.test_number + "/step_loss.txt")
	stepdiff = get_data("results/" + args.test_number + "/step_diff.txt")

	#steploss2 = get_data("results/002/step_loss.txt")
	#stepdiff2 = get_data("results/002/step_diff.txt")

	#steploss_val = get_data("results/" + args.test_number + "/step_val_loss.txt")
	#stepdiff_val = get_data("results/" + args.test_number + "/step_val_diff.txt")

	#multiple_plot_results(avgloss,avgdiff,steploss,stepdiff)
	
	plot_results(steploss, "Training Stepwise Loss", True, True)
	plot_results(stepdiff, "Training Stepwise Differences", True)

	#plot_results(steploss2, "Training Stepwise Loss", True, True)
	#plot_results(stepdiff2, "Training Stepwise Differences", True)

	#plot_results(steploss_val, "Validation Stepwise Loss", True, True)
	#plot_results(stepdiff_val, "Validation Stepwise Differences", True)
	plt.show()
