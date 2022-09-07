## FOR HEADLINES 
#  FOR CONSOLE CODES

## IMPORT LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

matplotlib.pyplot.show()
matplotlib.rcParams['figure.figsize'] = (12,8) # Adjust The Configuration Of Plots

## READ IN THE DATA

Movies_Table = pd.read_csv('D:\Data\movies.csv')

## SEARCHING FOR MISSING DATA

for col in Movies_Table.columns:
    missing_percent = np.mean(Movies_Table[col].isnull())
    print('{} - {}'.format(col,missing_percent))

## COLUMNS DATA TYPES

Movies_Table.dtypes

## TURNED ALL NULLS INTO 0

Movies_Table = Movies_Table.fillna('0')

## CONVERT BUDGET AND GROSS FLOAT TO INT

Movies_Table['budget'] = Movies_Table['budget'].astype('int64')
Movies_Table['gross'] = Movies_Table['gross'].astype('int64')

## SEPARETE YEAR FROM THE RELEASED COLUMN TO GET CORRECT YEAR

Movies_Table['correct_year'] = Movies_Table['released'].astype('string').str[:4]

## ORDER BY GROSS (Not For PyCharm)

#Movies_Table.sort_values(by=['gross'], inplace=False, ascending=False)

## DISPLAY ALL ROWS (Not For PyCharm)

#pd.set_option('display.max_rows',None)

## DROP DUPLICATES

#Movies_Table['company'].drop_duplicates().sort_values(ascending=False)

## CORRELATION WITH BUDGET-GROSS

plt.scatter(x=Movies_Table['budget'],y=Movies_Table['gross'])
plt.title('Budget-Gross')
plt.xlabel('Budget(Hundred Million)')
plt.ylabel('Gross(Billion)')
#plt.show()

## REGPLOT WITH SEABORN (BUDGET-GROSS)

sns.regplot(x='budget', y='gross', data = Movies_Table,
            scatter_kws={"color":"red"}, line_kws={"color":"blue"})
#plt.show()

## CORRELATION RATIO

print(Movies_Table.corr(method = 'spearman'))

## CORRELATION MATRIX

correlation_matrix = Movies_Table.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

## CORRELATION WITH STRING VALUES

Movies_Table_numerized = Movies_Table

for col_name in Movies_Table_numerized.columns:
    if(Movies_Table_numerized[col_name].dtype == 'object'):
        Movies_Table_numerized[col_name] = Movies_Table_numerized[col_name].astype('category')
        Movies_Table_numerized[col_name] = Movies_Table_numerized[col_name].cat.codes
print(Movies_Table_numerized)

correlation_matrix = Movies_Table_numerized.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()

## CORRELATION UNSTACKED

correlation_mat = Movies_Table_numerized.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs

sorted_pairs = corr_pairs.sort_values()
sorted_pairs

sorted_pairs[(sorted_pairs) > 0.5]

## BUDGET AND VOTES HAVE HIGH INFLUENCE ON MOVIE'S GROSS ##
