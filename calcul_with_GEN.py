from transformers import pipeline
import os
import pandas as pd
import csv
import coloralf as c
import numpy as np
from tqdm import tqdm

from pgfn import predictGEN_fined_tune as PGFN
			



def all_calcul_GEN(folder, case, desc, force=False):

	predicteur = PGFN(desc)

	c.fm(f"\n======== ALL calcul ======== FOR model {desc}")
	if force : c.fm(f"-> with force")

	for langage in os.listdir(f"{folder}/{case}"):
		
			c.fy(f"Run {case} * {langage}")

			if f"predictions_{predicteur.desc}.csv" not in os.listdir(f"{folder}/{case}/{langage}"):

				cases = pd.read_csv(f"{folder}/{case}/{langage}/cases_final.csv", index_col=0)


				predictions, probability = predicteur.run(folder, case, langage)

				df = cases.copy()
				df['predictions'] = predictions
				df['probability'] = probability

				# os.remove(f"{folder}/{case}/{langage}/predictions_opt.csv")
				df.to_csv(f"{folder}/{case}/{langage}/predictions_{predicteur.desc}.csv")