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

def compare_plots(data1, data2, legen, titl, ax1, ax2, fignr):
	CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

	plt.figure(fignr)
	plt.plot(data2[:,0], data2[:,1])
	for i in range(len(data1)):
		plt.plot(data1[i][:,0], data1[i][:,1], color=CB_color_cycle[i], linestyle='--')

	plt.legend(legen)
	plt.title(titl)
	plt.grid()
	plt.xlabel(ax1)
	plt.ylabel(ax2)

	plt.figure(fignr+1)
	plt.plot(data2[:,0], data2[:,2])
	for i in range(len(data1)):
		plt.plot(data1[i][:,0], data1[i][:,2], color=CB_color_cycle[i],linestyle='--')

	plt.legend(legen)
	plt.title(titl)
	plt.grid()
	plt.xlabel(ax1)
	plt.ylabel("Total displacement / %")
	#plt.ylabel("Voxels")
	plt.show()

def plot_results(data, length, title, fig, legds, xaxis, yaxis, log=False, width=1):
	plt.figure(fig)
	x = []
	y = []
	z = []
	min = []

	for i in range(length):
		x.append(data[i][0])
		y.append(data[i][1])
		z.append(data[i][2])
		min.append(data[i][3])

	plt.plot(x, y, linewidth=width)
	plt.plot(x, z, linewidth=width)
	plt.plot(x, min, linewidth=width)
	plt.title(title)
	plt.legend(legds)
	plt.grid()
	plt.xlabel(xaxis)
	plt.ylabel(yaxis)

	if log:
		plt.yscale('log')

def parse_args():
	"""Parse command line argument."""
	parser = argparse.ArgumentParser("Train segmention model on 3D structures.")
	parser.add_argument("model_number", help="Choose which model to display results from")
	parser.add_argument("test_number", help="logs the result files to specific runs")
	parser.add_argument("mode", help="Look at the training or test results")
	parser.add_argument("type", help="Look at the Supervised or Reinforcement Learning")

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	#avgloss = get_data("results/" + args.test_number + "/avg_train_loss.txt")
	#avgdiff = get_data("results/" + args.test_number + "/avg_train_diff.txt")
	legds = []
	if args.type == 's':
		if args.mode == 'train':
			print("Choose test number for model 0")
			test_nr0 = input()
			print("Choose test number for model 1")
			test_nr1 = input()
			print("Choose test number for model 2")
			test_nr2 = input()

			folder_0 = "results_0/" + test_nr0
			legds.append("Model 0")
			folder_1 = "results_1/" + test_nr1
			legds.append("Model 1")
			folder_2 = "results_2/" + test_nr2
			legds.append("Model 2")

			file_name1 = "/step_loss.txt"
			file_name2 = "/step_diff.txt"
			steploss = get_data(folder_0 + file_name1)
			stepdiff = get_data(folder_0 + file_name2)
			plot_results(steploss, "Training Stepwise Loss", 1, legds, "Loss", log=True)
			plot_results(stepdiff, "Training Stepwise Differences", 2, legds, "Difference", log=True, width=0.75)

			steploss2 = get_data(folder_1 + file_name1)
			stepdiff2 = get_data(folder_1 + file_name2)
			plot_results(steploss2, "Training Stepwise Loss", 1, legds, True, "Epoch", "Loss", log=True)
			plot_results(stepdiff2, "Training Stepwise Differences", 2, legds, "Epoch", "Difference", log=True, width=0.75)

			steploss3 = get_data(folder_2 + file_name1)
			stepdiff3 = get_data(folder_2 + file_name2)
			plot_results(steploss3, "Training Stepwise Loss", 1, legds, True, "Epoch", "Loss", log=True)
			plot_results(stepdiff3, "Training Stepwise Differences", 2, legds, "Epoch", "Difference", log=True, width=0.75)
		else:
			folder = "test_results/model" + args.model_number + "_"
			threshold1 = "50"
			threshold2 = "51"
			threshold3 = "55"
			threshold4 = "60"
			threshold5 = "70"
			file_name1 = "/generative_loss.txt"
			file_name2 = "/3Dconvmodel_loss.txt"
			#data1 = get_data(folder + args.test_number + "_" + threshold1 + file_name2)
			data2 = get_data(folder + args.test_number + "_" + threshold2 + file_name2)
			data3 = get_data(folder + args.test_number + "_" + threshold3 + file_name2)
			data4 = get_data(folder + args.test_number + "_" + threshold4 + file_name2)
			data5 = get_data(folder + args.test_number + "_" + threshold5 + file_name2)
			datagen = get_data(folder + args.test_number + "_" + threshold5 + file_name1)
			legen = ['Generative max bending', '3Dconv thres ' + threshold2, '3Dconv thres ' + threshold3, '3Dconv thres ' + threshold4, '3Dconv thres ' + threshold5, ]
			titl = 'Generative vs 3Dconv'
			ax1 = 'Generations'
			ax2 = 'Max displacement / %'
			compare_plots([data2, data3, data4, data5], datagen, legen, titl, ax1, ax2, 1)

	else:
		folder = "rl2_data/" + args.model_number
		data_con = []
		for i in range(int(args.test_number)):
			data = get_data(folder + "/Episode-stats_" + str(i) + ".txt")
			data_con.append(data)
		print(data_con[0][1])
		title = "Reinforcement Learning episode performance"
		fig = 1
		legds = ["Max", "Mean", "Min"]
		xaxis = "Epoch"
		yaxis = "Reward"
		plot_results(data_con, int(args.test_number), title, fig, legds, xaxis, yaxis, log=False, width=1)

	plt.show()
