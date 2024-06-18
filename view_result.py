import os
import shutil
import pandas as pd
import csv
import coloralf as c
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def view(folder, case, langage, desc):

	c.fy(f"Make {case} * {langage}...")

	##### Preparation

	folder_result = f"{folder}_RESULT_{desc}"

	# Creation of the result folder if not exist
	if f"{case}_RESULT" not in os.listdir(folder_result):
		os.mkdir(f"{folder_result}/{case}_RESULT")

	# Recuperation of prediction for given case & langage
	df = pd.read_csv(f"{folder}/{case}/{langage}/predictions_{desc}.csv")

	# Define the list of identities
	idents = list(pd.unique(df.target_ident))
	if 'neutre' in idents:
		idents.pop(idents.index('neutre'))

	# Create dataframe for stock result
	INDEX = ['0_Numbers', '1_ACC', '1_AUC', '2_H-F1', '2_NH-F1', '2_MacF1', '2_F1-AUC']

	INDEX += [f"tp_{ident}" for ident in idents]
	INDEX += [f"tn_{ident}" for ident in idents]
	INDEX += [f"fp_{ident}" for ident in idents]
	INDEX += [f"fn_{ident}" for ident in idents]

	INDEX += [f"ACC_{ident}" for ident in idents]
	INDEX += [f"MM_{ident}" for ident in idents]
	INDEX += [f"OP_{ident}" for ident in idents]
	INDEX += [f"AUC_{ident}" for ident in idents]
	INDEX += [f"MacF1_{ident}" for ident in idents]
	INDEX += [f"F1-AUC_{ident}" for ident in idents]

	dfn = pd.DataFrame(columns=[langage], index=INDEX)

	# Create empty data list for push the results
	data_result = np.zeros(len(INDEX))

	# Extraction of label & predictions
	gold = np.array(df['label_gold'])
	pred = np.array(df['predictions'])




	##### Calcul ... (_nh is for non-hatefull values)

	# Calculation of TP, TN, FP, FN
	tp, tp_nh = np.sum((gold == 'hateful') & (pred == 'hateful')), np.sum((gold == 'non-hateful') & (pred == 'non-hateful'))
	fp, fp_nh = np.sum((gold != 'hateful') & (pred == 'hateful')), np.sum((gold != 'non-hateful') & (pred == 'non-hateful'))
	tn, tn_nh = np.sum((gold != 'hateful') & (pred != 'hateful')), np.sum((gold != 'non-hateful') & (pred != 'non-hateful'))
	fn, fn_nh = np.sum((gold == 'hateful') & (pred != 'hateful')), np.sum((gold == 'non-hateful') & (pred != 'non-hateful'))

	# Calculation of all P & N
	p, p_nh = np.sum(gold == 'hateful'), np.sum(gold == 'non-hateful')
	n, p_nh = np.sum(gold != 'hateful'), np.sum(gold != 'non-hateful')

	# Precision & Recall
	ppv = 0 if tp==0 else tp / (tp + fp)
	tpr = 0 if tp==0 else tp / (tp + fn)
	ppv_nh = 0 if tp_nh==0 else tp_nh / (tp_nh + fp_nh)
	tpr_nh = 0 if tp_nh==0 else tp_nh / (tp_nh + fn_nh)

	# F1 score, accuracy, MCC
	f1 = 0 if (ppv + tpr) == 0 else 2 * (ppv * tpr) / (ppv + tpr)
	f1_nh = 0 if (ppv_nh + tpr_nh) == 0 else 2 * (ppv_nh * tpr_nh) / (ppv_nh + tpr_nh)
	acc = (tp + tn) / (p + n)

	# AUC
	true_label = gold == 'hateful'
	pred_proba = np.array(df['probability'])
	auc = roc_auc_score(true_label, pred_proba)
	
	# Push this calculus in the data
	data_result[INDEX.index(f"0_Numbers")] = p + n
	data_result[INDEX.index(f"1_ACC")]     = np.round(acc*100, 2)
	data_result[INDEX.index(f"1_AUC")]     = np.round(auc*100, 2)
	data_result[INDEX.index(f"2_H-F1")]    = np.round(f1*100, 2)
	data_result[INDEX.index(f"2_NH-F1")]   = np.round(f1_nh*100, 2)
	data_result[INDEX.index(f"2_MacF1")]   = np.round((f1+f1_nh)/2*100, 2)
	data_result[INDEX.index(f"2_F1-AUC")]   = np.round((f1+f1_nh+2*auc)/4*100, 2)

	


	##### Calcul for each identities

	for i, ident in enumerate(idents):

		tp = df[(df.target_ident==ident) & (df['label_gold']=='hateful') & (df['predictions']=='hateful')].shape[0]
		tn = df[(df.target_ident==ident) & (df['label_gold']!='hateful') & (df['predictions']!='hateful')].shape[0]
		fp = df[(df.target_ident==ident) & (df['label_gold']!='hateful') & (df['predictions']=='hateful')].shape[0]
		fn = df[(df.target_ident==ident) & (df['label_gold']=='hateful') & (df['predictions']!='hateful')].shape[0]

		tp_nh = df[(df.target_ident==ident) & (df['label_gold'] == 'non-hateful') & (df['predictions'] == 'non-hateful')].shape[0]
		fp_nh = df[(df.target_ident==ident) & (df['label_gold'] != 'non-hateful') & (df['predictions'] == 'non-hateful')].shape[0]
		tn_nh = df[(df.target_ident==ident) & (df['label_gold'] != 'non-hateful') & (df['predictions'] != 'non-hateful')].shape[0]
		fn_nh = df[(df.target_ident==ident) & (df['label_gold'] == 'non-hateful') & (df['predictions'] != 'non-hateful')].shape[0]

		ppv = 0 if tp==0 else tp / (tp + fp)
		tpr = 0 if tp==0 else tp / (tp + fn)
		p = tp + fn
		n = fp + tn
		n_tot = p + n

		ppv_nh = 0 if tp_nh==0 else tp_nh / (tp_nh + fp_nh)
		tpr_nh = 0 if tp_nh==0 else tp_nh / (tp_nh + fn_nh)

		id_acc = (tp + tn) / n_tot
		id_mm = 1 - fn / p
		id_op = 1 - fp / n

		id_f1 = 0 if (ppv + tpr) == 0 else 2 * (ppv * tpr) / (ppv + tpr)
		id_f1_nh = 0 if (ppv_nh + tpr_nh) == 0 else 2 * (ppv_nh * tpr_nh) / (ppv_nh + tpr_nh)

		sub_df = df[(df.target_ident==ident)]
		true_label = list(sub_df['label_gold']=='hateful')
		pred_proba = list(sub_df['probability'])

		id_auc = roc_auc_score(true_label, pred_proba)
		id_f1auc = (2*id_auc + id_f1 + id_f1_nh)/4

		data_result[INDEX.index(f"tp_{ident}")] = tp
		data_result[INDEX.index(f"tn_{ident}")] = tn
		data_result[INDEX.index(f"fp_{ident}")] = fp
		data_result[INDEX.index(f"fn_{ident}")] = fn

		data_result[INDEX.index(f"ACC_{ident}")] = np.round(id_acc*100, 2) # Accuracy
		data_result[INDEX.index(f"MM_{ident}")] = np.round(id_mm*100, 2)   # Recall
		data_result[INDEX.index(f"OP_{ident}")] = np.round(id_op*100, 2)   # Over Protection (selectivity / specificity)
		data_result[INDEX.index(f"AUC_{ident}")] = np.round(id_auc*100, 2) # AUC
		data_result[INDEX.index(f"MacF1_{ident}")] = np.round((id_f1+id_f1_nh)/2*100, 2)   # Macro F1 score
		data_result[INDEX.index(f"F1-AUC_{ident}")] = np.round(id_f1auc*100, 2)   # Ajuted Macro F1 score AUC




	##### Save the result

	dfn[langage] = data_result

	# If a CSV doest not exist for our case, I save directly the DataFrame to CSV
	if f"{case}_RESULT.csv" not in os.listdir(f"{folder_result}/{case}_RESULT"):
		
		c.fly(f" -> NEW : Create csv...")
		dfn.to_csv(f"{folder_result}/{case}_RESULT/{case}_RESULT.csv")

	else:
		# If a CSV already exist, I read it
		dfnINIT = pd.read_csv(f"{folder_result}/{case}_RESULT/{case}_RESULT.csv", index_col=0)

		if langage in dfnINIT.columns:
			# If the langage already exist, we update the csv
			c.fly(f" -> Update csv...")
			dfnINIT.update(dfn)
			
			dfnINIT.to_csv(f"{folder_result}/{case}_RESULT/{case}_RESULT.csv")
		else:
			# If the langage not in the existing csv, I complete it
			c.fly(f" -> Complete csv...")
			merged_df = dfnINIT.merge(dfn, how='outer', left_index=True, right_index=True).sort_index()
			merged_df.to_csv(f"{folder_result}/{case}_RESULT/{case}_RESULT.csv")



def GMB(scor, p=-5):

	if len(scor) == 0 : return 0
	else : return (1/len(scor) * np.sum(scor**p)) ** (1/p)



def defineRange(tab1, tab2):

	vmin = min(np.min(tab1[np.isnan(tab1)==False]), np.min(tab2[np.isnan(tab2)==False]))
	vmax = max(np.max(tab1[np.isnan(tab1)==False]), np.max(tab2[np.isnan(tab2)==False]))
	vabs = max(np.abs(vmin), np.abs(vmax))

	if   vabs > 100 : return -100, 100
	elif vabs > 50  : return -50, 50
	elif vabs > 20  : return -20, 20
	elif vabs > 15  : return -15, 15
	else : return -10, 10 



def make_graph(folder, case, cmap=['RdYlGn', 'PRGn', 'RdYlGn'], fmt='.1f', linewidths=1.0, folder_result=""):

	c.fy(f"Make graph {case} ...")

	# List of metrics
	metrics = ["Accuracy", "Recall", "Over Detection", "AUC", "MacF1", "F1-AUC"]

	# Extract data for the case
	df = pd.read_csv(f"{folder_result}/{case}_RESULT/{case}_RESULT.csv", index_col=0)

	# separate indice for each categorie
	ind = list(df.index)
	desc = [indi for indi in ind if indi[0] in ['1', '2', '3', '4', '5']]

	tp = [indi for indi in ind if 'tp_' in indi]
	tn = [indi for indi in ind if 'tn_' in indi]
	fp = [indi for indi in ind if 'fp_' in indi]
	fn = [indi for indi in ind if 'fn_' in indi]

	acc = [indi for indi in ind if 'ACC_' in indi]
	mm = [indi for indi in ind if 'MM_' in indi]
	op = [indi for indi in ind if 'OP_' in indi]
	auc = [indi for indi in ind if ('AUC_' in indi and 'F1-' not in indi)]
	f1 = [indi for indi in ind if 'MacF1_' in indi]
	f1auc = [indi for indi in ind if 'F1-AUC_' in indi]

	# create dict for containt global description for identities (ID) and langage (LA)
	# data_global_ID = dict()
	# data_global_LA = dict()
	# with they index :
	# global_index_ID = [acci.replace('ACC_', '') for acci in acc]
	# global_index_LA = list(df.columns)

	data_prec = [acc, mm, op, auc, f1, f1auc]




	### GRAPH Bias

	# list for tested p-values
	pGMB = [1, -3, -5, -8]
	# dicts for containt GMB value for each p
	rGMBauc_ID = dict()
	rGMBauc_LA = dict()
	
	# remember of mean of bias & std of bias
	FINAL = dict()

	for target in [acci.replace('ACC_', '') for acci in acc]:
		FINAL[target] = f"Target {target}"
	for langue in list(df.columns):
		FINAL[langue] = f"Langage {langue}"

	for pGMBi in pGMB:

		c.fly(f"Make GMB p={pGMBi}")

		rGMBauc_ID[f"p={pGMBi}"] = list()
		rGMBauc_LA[f"p={pGMBi}"] = list()

		# create dict for containt global description for identities (ID) and langage (LA)
		data_global_ID = dict()
		data_global_LA = dict()
		# with they index :
		global_index_ID = [acci.replace('ACC_', '') for acci in acc]
		global_index_LA = list(df.columns)



		# we create the folder for pGMBi
		if f"GMB p={pGMBi}" not in os.listdir(f"{folder_result}/{case}_RESULT"):
			os.mkdir(f"{folder_result}/{case}_RESULT/GMB p={pGMBi}")

		# we iterate on metrics
		for prec, title in zip(data_prec, metrics):

			data_global_ID[title] = list()
			data_global_LA[title] = list()

			pNaN, p = df.loc[prec].values.copy(), df.loc[prec].values.copy()
			p[np.isnan(pNaN)] = 0.0
			whereNaN = np.isnan(pNaN) != True



			## Matrice des GMB sur les langue
			gmb_ident = np.zeros_like(p)

			for i in range(p.shape[1]):
				ps = p[:, i][p[:, i] != 0]

				here_gmb = GMB(ps, p=pGMBi)

				gmb_ident[:, i] = here_gmb
				data_global_LA[title].append(here_gmb)
				if title == 'F1-AUC': 
					rGMBauc_LA[f"p={pGMBi}"].append(here_gmb)
					line = (ps - here_gmb) / here_gmb * 100
					variance = 1/len(ps) * np.sum([(psi-here_gmb)**2 for psi in ps])

					FINAL[global_index_LA[i]] += f",{np.sum(np.abs(line)):.2f},{np.std(line):.2f},{np.sqrt(variance):.2f}"

			bias_ident = df.loc[prec].values.copy()
			bias_ident[whereNaN] = (p[whereNaN] - gmb_ident[whereNaN]) / gmb_ident[whereNaN] * 100

			# print(bias_ident)


			## Matrice des GMB sur les targets
			gmb_lang = np.zeros_like(p)

			for j in range(p.shape[0]):
				ps = p[j][p[j] != 0]

				here_gmb = GMB(ps, p=pGMBi)

				gmb_lang[j] = here_gmb
				data_global_ID[title].append(here_gmb)
				if title == 'F1-AUC': 
					rGMBauc_ID[f"p={pGMBi}"].append(here_gmb)
					line = (ps - here_gmb) / here_gmb * 100
					variance = 1/len(ps) * np.sum([(psi-here_gmb)**2 for psi in ps])

					FINAL[global_index_ID[j]] += f",{np.sum(np.abs(line)):.2f},{np.std(line):.2f},{np.sqrt(variance):.2f}"

			bias_lang = df.loc[prec].values.copy()
			bias_lang[whereNaN] = (p[whereNaN] - gmb_lang[whereNaN]) / gmb_lang[whereNaN] * 100


			# BIAS GRAPH
			plt.figure(figsize=(16, 8))

			plt.subplot(131)
			plt.title(f"{title} (%) [for GMB p={pGMBi}]")
			sns.heatmap(df.loc[prec], cmap=cmap[0], annot=True, fmt=fmt, linewidths=linewidths, linecolor='black')
			plt.xticks(rotation=25)
			plt.yticks(plt.yticks()[0], [preci.replace('PREC_', '') for preci in prec], rotation=-15)

			
			vmin, vmax = defineRange(bias_ident, bias_lang)



			plt.subplot(132)
			plt.title(f"Bias between targets for each langage (%)")
			sns.heatmap(bias_ident, cmap=cmap[1], annot=True, fmt=fmt, vmin=vmin, vmax=vmax, linewidths=linewidths, linecolor='black')
			plt.xticks(plt.xticks()[0], df.columns, rotation=25)
			plt.yticks([])

			plt.subplot(133)
			plt.title(f"Bias between langage for each targets (%)")
			sns.heatmap(bias_lang, cmap=cmap[1], annot=True, fmt=fmt, vmin=vmin, vmax=vmax, linewidths=linewidths, linecolor='black')
			plt.xticks(plt.xticks()[0], df.columns, rotation=25)
			plt.yticks([])

			plt.savefig(f"{folder_result}/{case}_RESULT/GMB p={pGMBi}/{case}_GRAPH {title}.png")
			plt.close()




			### GRAPH Global

			df_global_ID = pd.DataFrame(data_global_ID, index=global_index_ID)
			df_global_LA = pd.DataFrame(data_global_LA, index=global_index_LA)

			plt.figure(figsize=(16, 8))

			plt.subplot(121)
			plt.title(f"GMB (p={pGMBi}) for targets (%)")
			sns.heatmap(df_global_ID, cmap=cmap[2], annot=True, fmt=fmt, vmin=25, vmax=90, linewidths=linewidths, linecolor='black')
			plt.xticks(rotation=25)
			plt.yticks(rotation=-15)

			plt.subplot(122)
			plt.title(f"GMB for langage (%)")
			sns.heatmap(df_global_LA, cmap=cmap[2], annot=True, fmt=fmt, vmin=25, vmax=90, linewidths=linewidths, linecolor='black')
			plt.xticks(rotation=25)
			plt.yticks(rotation=-15)

			plt.savefig(f"{folder_result}/{case}_RESULT/GMB p={pGMBi}/{case}_GRAPH_GLOBAL.png")
			plt.close()


	### Save FINAL

	c.fly(f"Save FINAL...")

	with open(f"{folder_result}/{case}_RESULT/FINAL bias result.csv", "w") as f:

		f.write(f"pValue")
		for pi in pGMB : f.write(f",p={pi}_Sum,p={pi}_STD,p={pi}_STDp")
		f.write(f"\n")

		for key, val in FINAL.items():
			f.write(val + "\n")





	### Resume all p GMB AUC

	c.fly(f"Resume all p GMB AUC...")

	df_rid = pd.DataFrame(rGMBauc_ID, index=global_index_ID)
	df_rla = pd.DataFrame(rGMBauc_LA, index=global_index_LA)

	plt.figure(figsize=(16, 8))

	plt.subplot(121)
	plt.title(f"GMB F1-AUC for targets (%)")
	sns.heatmap(df_rid, cmap=cmap[2], annot=True, fmt=fmt, linewidths=linewidths, linecolor='black')
	plt.xticks(rotation=25)
	plt.yticks(rotation=-15)

	plt.subplot(122)
	plt.title(f"GMB F1-AUC for langage (%)")
	sns.heatmap(df_rla, cmap=cmap[2], annot=True, fmt=fmt, linewidths=linewidths, linecolor='black')
	plt.xticks(rotation=25)
	plt.yticks(rotation=-15)

	plt.savefig(f"{folder_result}/{case}_RESULT/{case}_GRAPH_GMB_F1-AUC.png")
	plt.close()




	### GRAPH Desciption

	c.fly(f"Make Description graph...")
	plt.figure(figsize=(16, 8))
	sns.heatmap(df.loc[desc], cmap=cmap[2], annot=True, fmt=fmt, linewidths=linewidths, linecolor='black')
	plt.xticks(rotation=25)
	plt.title(f"Caractéristique (%)")
	plt.savefig(f"{folder_result}/{case}_RESULT/{case}_GRAPH_DESC.png")
	plt.close()





	### GRAPH RESUME METRICS -> ["Accuracy", "Over Protection", "AUCbin", "AUC", "F1"]

	c.fly(f"Make Resume metrics graph...")
	plt.figure(figsize=(16, 8))
	vmin, vmax = 50, 80
	select_metrics = ["Macro F1", "AUC", "F1-AUC"]
	super_ = [f1, auc, f1auc]

	for i, (prec, title) in enumerate(zip(super_, select_metrics)):

		if i == len(select_metrics)-1 : cbar = True
		else : cbar = False

		nsub = 100 + 10*len(select_metrics) + i + 1
		plt.subplot(nsub)
		plt.title(f"{title} (%)")
		sns.heatmap(df.loc[prec], cmap=cmap[0], annot=True, fmt='.0f', linewidths=linewidths, linecolor='black', cbar=False)
		plt.xticks(rotation=25)

		if i == 0 : plt.yticks(plt.yticks()[0], [preci.split('_')[-1] for preci in prec], rotation=-15)
		else : plt.yticks([])

	plt.savefig(f"{folder_result}/{case}_RESULT/{case}_GRAPH RESUME METRICS.png")
	plt.close()





	### GRAPH NB VAL

	if f"confusion" not in os.listdir(f"{folder_result}/{case}_RESULT/"):
		os.mkdir(f"{folder_result}/{case}_RESULT/confusion")


	for ii, (prec, title) in enumerate(zip(data_prec, metrics)):

		c.fly(f"Make nb Val metric {title} graph...")

		plt.figure(figsize=(16, 8))

		plt.subplot(121)
		plt.title(f"{title} (%)")
		sns.heatmap(df.loc[prec], cmap=cmap[0], annot=True, fmt='.1f', linewidths=linewidths, linecolor='black', cbar=False)
		plt.xticks(rotation=25)
		plt.yticks(plt.yticks()[0], [preci.replace('PREC_', '') for preci in prec], rotation=-15)


		plt.subplot(122)
		plt.title(f"Numbers of TP , FP / FN , TN")
		sns.heatmap(df.loc[prec], cmap=cmap[0], linewidths=linewidths, linecolor='black', cbar=True)

		n, m = df.loc[prec].shape

		val_tp = df.loc[tp].values
		val_tn = df.loc[tn].values
		val_fp = df.loc[fp].values
		val_fn = df.loc[fn].values

		for i in range(n):
			for j in range(m):
				x = j
				y = i
				if not np.isnan(val_tp[i, j]):

					plt.annotate(f'{val_tp[i, j]:.0f}', (x + 0.25, y + 0.25), color='green', ha='center', va='center', fontsize=8, fontweight='bold',
						bbox=dict(boxstyle="round,pad=0.2", fc='white', ec="black", lw=1))
					plt.annotate(f'{val_tn[i, j]:.0f}', (x + 0.75, y + 0.75), color='green', ha='center', va='center', fontsize=8, fontweight='bold',
						bbox=dict(boxstyle="round,pad=0.2", fc='white', ec="black", lw=1))

					plt.annotate(f'{val_fn[i, j]:.0f}', (x + 0.25, y + 0.75), color='black', ha='center', va='center', fontsize=8,
						bbox=dict(boxstyle="round,pad=0.2", fc='white', ec="black", lw=1))
					plt.annotate(f'{val_fp[i, j]:.0f}', (x + 0.75, y + 0.25), color='black', ha='center', va='center', fontsize=8,
						bbox=dict(boxstyle="round,pad=0.2", fc='white', ec="black", lw=1))

		plt.xticks(rotation=25)
		plt.yticks([])

		plt.savefig(f"{folder_result}/{case}_RESULT/confusion/{title}_CONFUSION.png")
		plt.close()




	# graph confusion acc + over detection
	# data_prec = [acc, mcc, mm, op, auc, f1, f1auc]
	# metrics = ["Accuracy", "MCC", "Recall", "Over Detection", "AUC", "MacF1", "F1-AUC"]

	plt.figure(figsize=(16, 8))

	plt.subplot(121)
	plt.title(f"Accuracy (%)")
	sns.heatmap(df.loc[acc], cmap=cmap[0], annot=True, fmt='.1f', linewidths=linewidths, linecolor='black', cbar=False)
	plt.xticks(rotation=25)
	plt.yticks(plt.yticks()[0], [preci.split('_')[-1] for preci in acc], rotation=-15)

	plt.subplot(122)
	plt.title(f"Over Detection (%)")
	sns.heatmap(df.loc[op], cmap=cmap[0], annot=True, fmt='.1f', linewidths=linewidths, linecolor='black', cbar=False)
	plt.xticks(rotation=25)
	plt.yticks([])

	plt.savefig(f"{folder_result}/{case}_RESULT/confusion/new_ACC_OverD_CONFUSION.png")
	plt.close()







def all_view(folder, desc='default', onlyOne=None):

	c.fm(f"\n======== ALL view ========")

	for case in os.listdir(folder):

		if onlyOne is None or case == onlyOne:

			c.flm(f"Make {case} view...")

			folder_result = f"{folder}_RESULT_{desc}"
			if folder_result not in os.listdir('./'):
				os.mkdir(folder_result)

			if f"{case}_RESULT" in os.listdir(folder_result):
				c.flr(f"INFO : folder_result/{case}_RESULT delete for remake")
				shutil.rmtree(f"{folder_result}/{case}_RESULT")

			for langage in os.listdir(f"{folder}/{case}"):
				view(folder=folder, case=case, langage=langage, desc=desc)

			c.flm(f"Make {case} graph...")
			make_graph(folder=folder, case=case, folder_result=folder_result)


