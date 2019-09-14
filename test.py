import pandas as pd
import os
df = pd.read_csv(os.path.join('C:/Users/long/PycharmProjects/PycharmProjects/bert','data/train.tsv'),
				 sep='\t', encoding='utf-8', error_bad_lines=False)
