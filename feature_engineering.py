###############################################################################
###############################################################################
###############################################################################
##
##
## feature_engineering.py
##
## @author: Matthew Cline
## @version: 20171113
##
## Description: Simple script to read in the data for the Kaggle challenge
##              that we are using for our machine learning final project.
##              The goal is to create some visualizations to help with
##              feature extraction.
##
##
###############################################################################
###############################################################################
###############################################################################

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)



######## READ IN THE DATA ########
data = pd.read_csv('train.csv') # Assumes that the file is in the pwd
print(data)

######## REPORT THE NUMBER OF POSITIVE AND NEGATIVES ########
print(data['target'].value_counts())

####### PAIR PLOTS TO SHOW THE POSSIBLE SEPARATING FEATURES #######
# Take a sample of 10000 points so that the plot will finish this year.
dataSample, dataRemainder = np.split(data, [1000])
dataSample, dataRemainder = np.split(dataSample, [10], axis = 1)
print(dataSample)
sns.pairplot(dataSample.drop("id", axis=1), hue='target')
plt.show()

# for i in range(2, data.shape[1]):


