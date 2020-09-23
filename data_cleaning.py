# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 20:06:30 2020

@author: bkorzen
"""

import pandas as pd
import math

df_original = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

df_original.head()
df_original.columns
df_original.info()
df_original.describe()


pd.options.display.max_columns = 10
df_original.select_dtypes(include='float64').describe()
df_original.Gender.value_counts()

df = pd.DataFrame()

# rounded synthetically generated numerical data to proper values
df['age'] = df_original['Age'].apply(lambda x: int(math.ceil(x)))
df['height_rounded'] = df_original['Height'].apply(lambda x: round(x, 2))
df['weight_rounded'] = df_original['Weight'].apply(lambda x: round(x, 2))

# rounded synthetically generated numerical data to proper categorical values
df['veges_freq'] = df_original['FCVC'].apply(lambda x: 'Never' if int(round(x,0)) == 1 else ('Sometimes' if int(round(x,0)) == 2 else ('Always')))
df['main_meals_num'] = df_original['NCP'].apply(lambda x: 'One/Two' if int(round(x,0)) <= 2 else ('Three' if int(round(x,0)) == 3 else ('More than 3')))
df['daily_water_consumption'] = df_original['CH2O'].apply(lambda x: '<1L' if int(round(x,0)) == 1 else ('1-2L' if int(round(x,0)) == 2 else ('>2L')))
df['physical_activity_freq'] = df_original['FAF'].apply(lambda x: '0 days' if int(round(x,0)) == 0 else ('1-2 days' if int(round(x,0)) == 1 else ('2-4 days' if int(round(x,0)) == 3 else ('4-5 days'))))
df['tech_devices_usage'] = df_original['TUE'].apply(lambda x: '0-2h' if int(round(x,0)) == 0 else ('3-5h' if int(round(x,0)) == 1 else ('>5h')))


# rename columns
df['high_kcal_food'] = df_original['FAVC']
df['transport_used'] = df_original['MTRANS']
df['snacks_consuming'] = df_original['CAEC']
df['smoking'] = df_original['SMOKE']
df['kcal_monitoring'] = df_original['SCC']
df['alcohol_consumption'] = df_original['CALC']
df['gender'] = df_original['Gender']

# create bmi param
df['bmi'] = df.apply(lambda x: x.weight_rounded / (2*x.height_rounded), axis=1)

df.describe()
df.info()

# df.to_csv('obesity_cleaned_improved.csv', index=False)

