#================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#									NAIVE BAYES
#----------------------------------------------------------------------------------------------------------------
#================================================================================================================

#Using Naive Bayes to classify emails as spam and non-spam

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas
import nltk
import re

#for plotting
plt.style.use('ggplot')

class CustomNB:
	
	def __init__(self):
		pass
	
	def create_freq_table(texts, labels=None, parse=False):
		pass
	
	def train(freq):
		pass
	
	def predict(text, prob_o_spam, prob_o_not_spam):
		pass

def main():
	
	# 0 means spam, 1 means not spam
	emails = {"0": ["Dear friend, win 1000$ cash right now!!.", " Webcams Day & Night - All LIVE - Webcams Contest",
			"Congratulations, you've won a free car!"], 
			"1": ["Hey, Can you redo the presentation and send it on by Friday?", "Thank you for the documents, I will revert as soon as possible",
			"Open the Pod bay doors, HAL." , " I'm sorry, Dave. I'm afraid I can't do that."]}
	t= []
	labl = []
		
	for k in emails:
		for mail in emails[k]:
			tokens = [word.lower() for sent in nltk.sent_tokenize(mail) for word in nltk.word_tokenize(sent)]
			filtered_tokens = []
			# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
			for token in tokens:
				if re.search('[a-zA-Z]', token):
					filtered_tokens.append(token)
			t.append(' '.join(filtered_tokens))
			labl.append(k)

	print(labl, t)
	nb = CustomNB()

if __name__ == "__main__":
	main()