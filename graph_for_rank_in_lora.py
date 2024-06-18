import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



def make_graph(basedesc, list_to_test, base=f"./data_RESULT", case=f"hatecheck_base", selected_metBias='p=-5_STDp', saveFolder='rankPlotVisu'):

	biasFile = f"FINAL bias result.csv"
	statFile = f"{case}_RESULT.csv"

	# basedesc = "mt0_hate_finedtune"
	# list_to_test = ['r2', 'r4', 'r8', 'r16', 'r32']

	if saveFolder not in os.listdir():
		os.mkdir(saveFolder)


	ranks = np.array([int(ri[1:].split('-')[0]) for ri in list_to_test])
	macf1 = np.zeros_like(ranks)
	biasTotal = np.zeros_like(ranks)
	biasLangue = np.zeros_like(ranks)
	biasTarget = np.zeros_like(ranks)


	for i, r in enumerate(list_to_test):

		folder = f"{base}_{basedesc}_{r}/{case}_RESULT"

		# bias part
		bias = pd.read_csv(f"{folder}/{biasFile}")

		listWhich = list(bias['pValue'])

		for j, which in enumerate(listWhich):

			biasTotal[i] += bias[selected_metBias].iloc[j]

			if "Target" in which:
				biasTarget[i] += bias[selected_metBias].iloc[j]
			elif "Langage" in which:
				biasLangue[i] += bias[selected_metBias].iloc[j]

		
		# macf1 part
		stat = pd.read_csv(f"{folder}/{statFile}", index_col=0)

		macf1[i] = np.sum(stat.loc["2_MacF1"]) / len(stat.loc["2_MacF1"])



	plt.plot(ranks, macf1, marker='x', linestyle='', color='b')
	plt.xlabel('Rank')
	plt.ylabel('Macro F1-score')
	plt.title('Macro F1-score along rank in LoRA')
	plt.savefig(f"{saveFolder}/macroF1.png")
	plt.close()

	plt.plot(ranks, biasTotal, marker='x', linestyle='', color='b')
	plt.xlabel('Rank')
	plt.ylabel('Sum of variance on BIAS (Total)')
	plt.title('Sum of variance on BIAS (Total), along rank in LoRA')
	plt.savefig(f"{saveFolder}/biasTotal.png")
	plt.close()

	plt.plot(ranks, biasLangue, marker='x', linestyle='', color='b')
	plt.xlabel('Rank')
	plt.ylabel('Sum of variance on BIAS (Language)')
	plt.title('Sum of variance on BIAS (Language), along rank in LoRA')
	plt.savefig(f"{saveFolder}/biasLangue.png")
	plt.close()

	plt.plot(ranks, biasTarget, marker='x', linestyle='', color='b')
	plt.xlabel('Rank')
	plt.ylabel('Sum of variance on BIAS (Target)')
	plt.title('Sum of variance on BIAS (Target), along rank in LoRA')
	plt.savefig(f"{saveFolder}/biasTarget.png")
	plt.close()




	for r, mf1, bt in zip(ranks, macf1, biasTotal):
		plt.scatter(mf1, bt, color='r', marker='.')
		plt.annotate(f"r={r}", xy=(mf1, bt), color='k', rotation=45,
					bbox=dict(boxstyle="round,pad=0.2", fc="lightgreen", ec="black", lw=1))

	plt.xlabel('Macro F1-score')
	plt.ylabel('Sum of variance on BIAS (Total)')
	plt.title('Sum of variance on BIAS (Target), in function of Macro F1-score')
	plt.savefig(f"{saveFolder}/mac-bias.png")
	plt.close()