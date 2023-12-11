import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("Sentences_AllAgree.txt")
df.head()