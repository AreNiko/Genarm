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

def compare_plots(data1, data2, legen, titl, ax1, ax2):
	plt.plot(data1[:,0], data1[:,1])
	plt.plot(data2[:,0], data2[:,1])
	plt.legend(legen)
	plt.title(titl)
	plt.xlabel(ax1)
	plt.ylabel(ax2)
	plt.show()

def plot_results(data, title, fig, legds, stepwise=False, loss=False):
	plt.figure(fig)
	plt.plot(data[:,0], data[:,1])
	plt.title(title)
	plt.legend(legds)
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
	parser.add_argument("mode", help="Look at the training or test results")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	#avgloss = get_data("results/" + args.test_number + "/avg_train_loss.txt")
	#avgdiff = get_data("results/" + args.test_number + "/avg_train_diff.txt")
	legds = []
	if args.mode == 'train':
		folder_0 = "results_0/" + args.test_number
		folder_1 = "results_1/" + args.test_number
		folder_2 = "results_2/" + args.test_number

		file_name1 = "/step_loss.txt"
		file_name2 = "/step_diff.txt"
		steploss = get_data(folder_0 + file_name1)
		stepdiff = get_data(folder_0 + file_name2)
		plot_results(steploss, "Training Stepwise Loss", 1, legds, True, True)
		plot_results(stepdiff, "Training Stepwise Differences", 2, legds, True)

		steploss2 = get_data(folder_1 + file_name1)
		stepdiff2 = get_data(folder_1 + file_name2)
		plot_results(steploss2, "Training Stepwise Loss", 1, legds, True, True)
		plot_results(stepdiff2, "Training Stepwise Differences", 2, legds, True)

		steploss3 = get_data(folder_2 + file_name1)
		stepdiff3 = get_data(folder_2 + file_name2)
		plot_results(steploss3, "Training Stepwise Loss", 1, legds, True, True)
		plot_results(stepdiff3, "Training Stepwise Differences", 2, legds, True)
	else:
		folder = "test_results/"
		file_name1 = "/generative_loss.txt"
		file_name2 = "/3Dconvmodel_loss.txt"
		data1 = get_data(folder + args.test_number + file_name1)
		data2 = get_data(folder + args.test_number + file_name2)
		legen = ['Generative loss', '3Dconv loss']
		titl = 'Generative vs 3Dconv'
		ax1 = 'epoch'
		ax2 = 'loss'
		compare_plots(data1, data2, legen, titl, ax1, ax2)
	
	plt.show()
