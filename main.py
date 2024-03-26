# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
df = pd.read_excel('C:\licenta\pacienti.xlsx', sheet_name= 'b')
print(df.head())

