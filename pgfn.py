import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
	get_peft_config,
	get_peft_model,
	get_peft_model_state_dict,
	set_peft_model_state_dict,
	LoraConfig,
	PeftType,
	PrefixTuningConfig,
	PromptEncoderConfig,

)

import evaluate
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import coloralf as c
import numpy as np
import pandas as pd




class predictGEN_fined_tune():

	"""
	Little class for make predictions
	"""



	def __init__(self, desc):

		self.batch_size = 16
		self.model_name_or_path = "bigscience/mt0-small"
		self.device = "cuda"

		self.peft_model_id = f"Aexeos/{desc}"
		self.desc = desc
		self.config = PeftConfig.from_pretrained(self.peft_model_id)
		self.inference_model = AutoModelForSequenceClassification.from_pretrained(self.config.base_model_name_or_path)
		self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path)

		# Load the Lora model
		self.inference_model = PeftModel.from_pretrained(self.inference_model, self.peft_model_id)



	def create_datasets(self, folder, base, langue):
		"""
		Creation of the dataset object
		"""

		data_files = {"validation" : f"./{folder}/{base}/{langue}/cases_final_genFormat.csv"}

		datasets = load_dataset('csv', data_files=data_files, encoding='utf-8')

		if any(k in self.model_name_or_path for k in ("gpt", "opt", "bloom")):
			padding_side = "left"
		else:
			padding_side = "right"

		tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side=padding_side)
		if getattr(tokenizer, "pad_token_id") is None:
			tokenizer.pad_token_id = tokenizer.eos_token_id

		def tokenize_function(examples):
			# max_length=None => use the model max length (it's actually the default)
			outputs = tokenizer(examples["text"], truncation=True, max_length=None)
			return outputs

		tokenized_datasets = datasets.map(
			tokenize_function,
			batched=True,
			remove_columns=["text"],
		)

		# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
		tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

		def collate_fn(examples):
			return tokenizer.pad(examples, padding="longest", return_tensors="pt")

		# Instantiate dataloaders.
		eval_dataloader = DataLoader(tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=self.batch_size)

		return eval_dataloader



	def run(self, folder, base, langue):
		"""
		Run predictions
		"""

		# Creatino of "cases_final_genFormat" -> it's the format to give in the dataset object
		if "cases_final_genFormat.csv" not in os.listdir(f"./{folder}/{base}/{langue}"):

			cases = pd.read_csv(f"{folder}/{base}/{langue}/cases_final.csv", index_col=0)

			newdf = pd.DataFrame(columns=['text', 'label'])

			newdf['text'] = cases['test_case']

			label_gold = np.array(cases['label_gold'])

			label_new = np.zeros(len(label_gold))
			label_new[label_gold == 'hateful'] = 1

			newdf['label'] = label_new.astype(int)

			# os.remove(f"{folder}/{base}/{langue}/predictions_optFormat.csv")
			newdf.to_csv(f"{folder}/{base}/{langue}/cases_final_genFormat.csv", index=False)

		eval_dataloader = self.create_datasets(folder, base, langue)

		predi = list() 
		label = list()

		self.inference_model.to(self.device)
		self.inference_model.eval()
		for step, batch in enumerate(tqdm(eval_dataloader)):
			batch.to(self.device)
			with torch.no_grad():
				outputs = self.inference_model(**batch)

			# print(outputs.logits)
			predictions = outputs.logits.argmax(dim=-1)
			predictions, references = predictions, batch["labels"]

			predi += predictions.tolist()
			label += references.tolist()


		pred = np.array(predi)
		gold = np.array(label)

		return_pred = list()
		return_prob = list()

		for p in pred:

			if p == 0:

				return_pred.append('non-hateful')
				return_prob.append(0)

			else:

				return_pred.append('hateful')
				return_prob.append(1)

		return return_pred, return_prob