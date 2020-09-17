# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:06:30 2020

@author: bkorzen
"""

import pandas as pd
import math

df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

df.head()
df.columns
df.info()
df.describe()


pd.options.display.max_columns = 10
df.select_dtypes(include='float64').describe()
df.Gender.value_counts()


# yes/no attributes parsing
df['family_history_with_overweight_binary'] = df['family_history_with_overweight'].apply(lambda x: 1 if x == 'yes' else 0)
df['FAVC_binary'] = df['FAVC'].apply(lambda x: 1 if x == 'yes' else 0)
df['SMOKE_binary'] = df['SMOKE'].apply(lambda x: 1 if x == 'yes' else 0)
df['SCC_binary'] = df['SCC'].apply(lambda x: 1 if x == 'yes' else 0)


# rounded synthetically generated numerical data to proper values
df['age_int'] = df['Age'].apply(lambda x: int(math.ceil(x)))
df['height_rounded'] = df['Height'].apply(lambda x: round(x, 2))
df['weight_rounded'] = df['Weight'].apply(lambda x: round(x, 2))

# rounded synthetically generated numerical data to proper categorical values
df['FCVC_cat'] = df['FCVC'].apply(lambda x: 'Never' if int(round(x,0)) == 1 else ('Sometimes' if int(round(x,0)) == 2 else ('Always')))
df['NCP_cat'] = df['NCP'].apply(lambda x: 'One/Two' if int(round(x,0)) <= 2 else ('Three' if int(round(x,0)) == 3 else ('More than 3')))
df['CH2O_cat'] = df['CH2O'].apply(lambda x: '<1L' if int(round(x,0)) == 1 else ('1-2L' if int(round(x,0)) == 2 else ('>2L')))
df['FAF_cat'] = df['FAF'].apply(lambda x: '0 days' if int(round(x,0)) == 0 else ('1-2 days' if int(round(x,0)) == 1 else ('2-4 days' if int(round(x,0)) == 3 else ('4-5 days'))))
df['TUE_cat'] = df['TUE'].apply(lambda x: '0-2h' if int(round(x,0)) == 0 else ('3-5h' if int(round(x,0)) == 1 else ('>5h')))

# bmi
df['bmi'] = df.apply(lambda x: x.weight_rounded / (2*x.height_rounded), axis=1)

df.to_csv('obesity_cleaned.csv', index=False)

