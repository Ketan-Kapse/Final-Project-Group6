import pandas as pd
import chardet

file_path = '../.venv/Sentences_AllAgree.txt'

import pandas as pd

# Read the file with the appropriate encoding
with open(file_path, 'r', encoding='latin-1') as file:
    lines = file.readlines()

# Splitting each line into sentence and label and storing them in a DataFrame
data = [line.strip().split('@') for line in lines]
df = pd.DataFrame(data, columns=['Sentence', 'Label'])

# Displaying the first few rows of the DataFrame to verify the result
df.head()