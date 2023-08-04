import pandas as pd
import numpy as np

#Loading the CSV file into a DataFrame:
df = pd.read_csv('indian_food.csv')

# Printing the size, shape, and data types of the DataFrame:
print(f'size: {df.size}')
print(f'shape: {df.shape}')
print(f'data types of each column in the dataset:\n{df.dtypes}')

#Getting the total number of Indian dishes
x, y = df.shape
print("Total no of Indian dishes: ", x)

# Replacing empty spaces with NaN
df.replace(' ', np.nan, inplace=True)

#Selecting and printing numerical columns
df.select_dtypes(include=['float64', 'int64']).columns

# Printing the number of unique values in each column
for col in df.columns:
    print(col, df[col].nunique())

# Adding a new column for the total time
df['total time'] = df['prep_time'] + df['cook_time']
print(df['total time'].head(10))

# Finding the number of ingredients for each recipe
df['num_ingredients'] = df['ingredients'].apply(len)
print(df['num_ingredients'].head(10))