import pandas as pd
  
# Create empty series and check its data type
s1 = pd.Series()
# Check its type and print series
type(s1)

"""### Creating a series from array"""

import pandas as pd
import numpy as np

array = ['r', 'e', 'l', 'e', 'v','e','l']
# Convert the given array to pandas series and print the series
series = pd.Series(array)
print(series)

"""Please note that index is assigned by default if not specified. Default value of index will start from 0 till length of the series. In this case it is 6.

### Creating a series from array with index
"""

data = np.array(['r', 'e', 'l', 5, 'v','e','l'])
  
# Giving an index [1, 10, 20, 30, 40, 50, 60] to series
series = pd.Series(data, index = [101, 10, 20, 30, 40, 50, 60])
print(series)

"""### Creating a series from Lists"""

list_to_convert = ['r', 'e', 'l', 'e', 'v','e','l']   
# create series form a list
series = pd.Series(list_to_convert, index = ['o', 'o', 'ii', 'iii', 'iv', 'o', 'vi'])
print(series)
#repeated indices also allowed

list1 = ['aaaaaa', 'bbb', 'cccc', 'ddd']
ids = [10001, 10002, 10003, 10004]

s1 = pd.Series(list1, index = ids)

s1

s1[10003]

series['o']

"""### Creating a series from Dictionary"""

dict_to_convert = {'Monday' : 0,
                   'Tuesday' : 2,
                    'Wednesday' : 4}
   
# create series from dictionary
series = pd.Series(dict_to_convert)
   
print(series)

"""Here Dictionary keys are taken as index values.

## Creating pandas dataframe

####A pandas dataframe is just like a data table with multiple rows and columns and it can also be viewed as a combination of series

### Creating an empty dataframe
"""

# always treat a dataframe as a tabluar data format

## Creating an empty dataframe
df = pd.DataFrame()
print(df)
type(df)

"""### Creating a dataframe using list"""

# list of strings
sentence_token = ['Relevel', 'is', 'the', 'best',
            'platform', 'to', 'learn','python']
 
# Calling DataFrame constructor on list
# Syntax: pandas.DataFrame(data=None, index=None, columns=None)
df = pd.DataFrame(sentence_token, index = ['o', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii'], columns = ['string11'])
print(df)
print(type(df))

df['string11']['i']

print(df['string11']['o'])
print(type(df))
print(type(df['string11']))

"""### Creating a dataframe from dictionary from list"""

import pandas as pd
 
# initialise data of lists.
state_data = {'State':['MP', 'UP', 'DL', None], 'Code': [10,15,19,24]}
 
# Create DataFrame from state_data using Pandas
df_1 = pd.DataFrame(state_data)
 
# Print the output
print(df_1)

df_1['Code'][2]

import pandas as pd
pd.DataFrame({'State':'MP', 'capital': 'bhopal'}, index = [1, 2, 3])

# Create DataFrame with index value ['City 1', 'City 2', 'City 3', 'City 4']
df_2 = pd.DataFrame(data = state_data, index=['City 1', 'City 2', 'City 3', 'City 4'])
 
# Print the output.
print(df_2)

"""### Create pandas DataFrame from dictionary of numpy array


"""

# Create a numpy array
nparray = np.array(
    [['Amar', 'Akbar', 'Ram', 'Ravi'],
     [29, 25, 33, 24],
     ['PR', 'Marketing', 'IT', 'Finance']])

# Create a dictionary of nparray
dictionary_of_nparray = {
    'Name': nparray[0],
    'Age': nparray[1],
    'Department': nparray[2]}

# Create the DataFrame
df1 = pd.DataFrame(nparray)
print(df1)
print(df1.shape)

df2 = pd.DataFrame(dictionary_of_nparray)
print(df2)
print(df2.shape)

"""Dictionary keys will come as column name while converting into DataFrame.

### Create pandas DataFrame from list of lists
"""

# Create a list of lists
list_of_lists = [
    ['Amar', 29, 'PR'],
    ['Akbar', 25, 'Finance'],
    ['Ram', 33, 'Marketing'],
    ['Ravi', 24, 'IT']]

# Create the DataFrame from list of list
df3_1 = pd.DataFrame(list_of_lists)
df3_1

disct1 = {'col1':list_of_lists[:][0], 'col2':list_of_lists[1], 'col3':list_of_lists[2] }
df4 = pd.DataFrame(disct1)
print(df4)

# we dont want this format

# Create the DataFrame with column names ['Name', 'Age', 'Department']
df3_2 = pd.DataFrame(list_of_lists, columns = ['Name', 'Age', 'Department'])
df3_2

df3_2['Age']

"""Here column names (as a list) are additionally passed as data is list of list. While in the previous code we just passed data which came default column index as 0, 1, 2 etc

### Create pandas DataFrame from list of dictionaries
"""

# Create a list of dictionaries
list_of_dictionaries = [
    {'Name': 'Ravi', 'Age': 29, 'Department': 'HR'},
    {'Name': 'Akbar', 'Age': 25, 'Department': 'Finance'},
    {'Name': 'Hari', 'Age': 33, 'Department': 'Marketing'},
    {'Name': 'Ramesh', 'Age': 24, 'Department': 'IT'}]

# Create the DataFrame
df = pd.DataFrame(list_of_dictionaries, index = range(1, len(list_of_dictionaries)+1))
df

df.columns

"""### Create pandas Dataframe from dictionary of pandas Series"""

# Create Series
series1 = pd.Series(['Emma', 'Oliver', 'Harry', 'Sophia'])
series2 = pd.Series([29, 25, 33, 24])
series3 = pd.Series(['HR', 'Finance', 'Marketing', 'IT'])

# Create a dictionary of Series
dictionary_of_series = {'Name': series1, 'Age': series2, 'Department':series3}

# Create the DataFrame
df1 = pd.DataFrame([series1, series2, series3])
print(df1)

df2 = pd.DataFrame(dictionary_of_series)
df2

type([series1, series2, series3])

"""## Read CSV"""

# ## Mount Drive
# from google.colab import drive
# drive.mount('/content/drive')

# import pandas as pd

# ## Read Excel file
# df = pd.read_excel('Banking.xlsx', sheet_name = 'Sheet 1')

# df

import pandas as pd
# Reading iris.csv data from github
# Visit URL - https://gist.github.com/netj/8836201#file-iris-csv. Click on "Raw" to get below link

# reading csv file
df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
df

sample_df = pd.read_csv('/content/sample.csv') #you need to upload this file first, use the above iris file instead
sample_df

## Find out shape of the dataframe
print(df.shape)
print(df.ndim)
df.size

"""### Describe Dataframe

Following data used here describes the sepal length, speal width, petal length, petal width of a given iris flow
"""

## Use describe function to find descriptive stats of dataframe
df.describe()

"""# Writing a file in Pandas to CSV and Excel

## Writing a dataframe to Excel file
"""

#usually not used
# pip install xlsxwriter
  
# Create a Pandas dataframe from some data.
df = pd.DataFrame({'Sentence': ['Relevel', 'is', 'the', 'best',
                               'platform', 'to', 'get','jobs']})
  
# Create a Pandas Excel writer
# object using XlsxWriter as the engine.
writer = pd.ExcelWriter('pandasEx.xlsx', 
                   engine ='xlsxwriter')
  
# Write a dataframe to the worksheet.
df.to_excel(writer, sheet_name ='myfile')
  
# Close the Pandas Excel writer
# object and output the Excel file.
writer.save()

"""## Writing a pandas dataframe to CSV file"""

#used most often
# Define a dictionary containing data
data = {'Name':['Hari', 'Ronak', 'Harleen', 'Parveen'],
        'Gender':["M", "M", "F", "F"],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
        'Age':[24, 26, 21, 30]}

# Convert the dictionary into DataFrame 
df = pd.DataFrame(data)

# saving the dataframe
df.to_csv('myfile.csv')

# import pandas as pd 

# #put here url where nba.csv is uploaded on drive
# url = "https://drive.google.com/file/d/1i8FEIBy8SlaRGyA7w5pSauSDS-_zK6M8/view?usp=sharing"
# url='https://drive.google.com/uc?id=' + url.split('/')[-2]

# # making data frame 
# df = pd.read_csv(url, index_col ="Name") 

# df.head()

"""# Advanced Pandas Function"""

# import pandas as pd

# # Put here url where nba.csv is uploaded on drive
# url = "https://drive.google.com/file/d/1i8FEIBy8SlaRGyA7w5pSauSDS-_zK6M8/view?usp=sharing"
# url='https://drive.google.com/uc?id=' + url.split('/')[-2]

# # Making data frame from csv file
# data = pd.read_csv(url, index_col ="Name")

# print(data)

"""dataframe.info() method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage"""

## Find out info of Dataframe
data.info()

data.shape

data.ndim

data = pd.read_csv('nba.csv')

## Check number of null values in Dataframe
data.isna().sum()

"""## Dealing with Columns in Pandas"""

data.columns

# Define a dictionary containing data
data = {'Name':['Hari', 'Ronak', 'Harleen', 'Parveen'],
        'Gender':["M", "M", "F", "F"],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
        'Age':[24, 26, 21, 30]}
  
# Convert the dictionary into DataFrame 
df = pd.DataFrame(data)

# print name column
print(df.columns.tolist())
print(list(df.columns))

#same codes in last 2 lines

# Define a dictionary containing data
data = {'Name':['Hari', 'Ronak', 'Harleen', 'Parveen'],
        'Gender':["M", "M", "F", "F"],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
        'Age':[24, 26, 21, 30]}
  
# Convert the dictionary into DataFrame 
df = pd.DataFrame(data)
  
# print two columns i.e 'Name', 'address'
print(df[['Name', 'Age']])   ## If multiple columns then that has to be passed as list i.e one value

"""###  Adding column"""

# Define a dictionary containing data
data = {'Name':['Hari', 'Ronak', 'Harleen', 'Parveen'],
        'Gender':["M", "M", "F", "F"],
        'Age':[24, 26, 21, 30]}
  
# Convert the dictionary into DataFrame 
df = pd.DataFrame(data)
print(df)
# Declare a list that is to be converted into a column
address = ['Mumbai', 'Vellore', 'Pune', 'Agra']

## Add address column from address variable mentioned above
df['Address'] = address
print(df)

"""**Create a new column based on existing column**

Example : Given a Dataframe containing data about an Fruits sold, Create a new column called ‘Profits’, which is calculated from Cost, Selling price and number of Kgs sold.
"""

# importing pandas as pd
import pandas as pd

dict_for_df  = {'Fruits':['Mango', 'Banana', 'Watermelon', 'Grapes'],
                    'Number of Kgs sold':[5, 10, 8, 4],
                    'Cost per Kg':[500, 80, 100, 170],
                   'Selling price per kg': [700, 120, 150, 200]}

# Creating the DataFrame
df = pd.DataFrame(dict_for_df)
  
# Print the dataframe
df

# create a new column based on the following formula
# Number of Kgs sold * (Selling price per kg - Cost per Kg)
df['Profit'] = df['Number of Kgs sold'] * (df['Selling price per kg'] - df['Cost per Kg'])
  
# Print the DataFrame after 
# addition of new column
df

df1 = df.assign(profit = lambda x: x['Number of Kgs sold'] * (x['Selling price per kg'] - x['Cost per Kg']))
df1

"""### Lambda function in Pandas"""

# Applying lambda function to single column using Dataframe.assign()

# creating and initializing a list
values= [['Rohan',455],['Elvish',250],['Deepak',495],
         ['Soni',400],['Radhika',350],['Vansh',450]]
 
# creating a pandas dataframe
df = pd.DataFrame(values,columns=['Name','Total_Marks'])
 
# Applying lambda function to find
# percentage of 'Total_Marks' column
# using df.assign()
# Percentage = Total_Marks /500 * 100
df.assign(percent = lambda x: df['Total_Marks']/5)
 
# assign function disp;ays the answer and not saves it

"""### Deleting Column"""

# Define a dictionary containing data
data = {'Name':['Hari', 'Ronak', 'Harleen', 'Parveen'],
        'Gender':["M", "M", "F", "F"],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
        'Age':[24, 26, 21, 30]}

# Convert the dictionary into DataFrame 
df = pd.DataFrame(data)

# dropping Gender columns
fd1 = df.drop(['Age', 'Gender'], axis = 1)
print(fd1)

"""## Dealing with Rows in Pandas

### Deleting Rows
"""

# Define a dictionary containing data
data = {'Name':['Hari', 'Ronak', 'Harleen', 'Parveen'],
        'Gender':["M", "M", "F", "F"],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
        'Age':[24, 26, 21, 30]}

# Convert the dictionary into DataFrame 
df = pd.DataFrame(data)

# dropping passed values ["Avery Bradley", "John Holland"]

df.drop([0,1]) #by default axis=0
df.drop(range(0,2))

"""## Adding Rows

### Using Concat and Append

There are situations when we have related data spread across multiple files.

The data can be related to each other in different ways. How they are related and how completely we can join the data from the datasets will vary.

In this exercise we will consider different scenarios and show we might join the data. We will use csv files and in all cases the first step will be to read the datasets into a pandas Dataframe from where we will do the joining.
"""

# Define a dictionary containing data
data = {'Name':['Hari', 'Ronak', 'Harleen', 'Parveen'],
        'Gender':["M", "M", "F", "F"],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
        'Age':[24, 26, 21, 30]}

# Convert the dictionary into DataFrame 
df = pd.DataFrame(data)
data1 = pd.DataFrame({'Name': ['Prime', 'JOHN'], 'Gender': ['M','M'], 'Address': ['Pune', 'jkl'], 'Age':[25, 45]})

data2 = pd.DataFrame({'Name': ['yee', 'jane'], 'Gender': ['M','F'], 'Address': ['Pune', 'jkl'], 'Age':[12, 8]}, index=[6,7])

df.append([data1, data2])
#by default the index starts from 0

#Deprecated since version 1.4.0: Use concat() instead.

"""Pandas `dataframe.append()` function is used to append rows of other dataframe to the end of the given dataframe, returning a new dataframe object. Columns not in the original dataframes are added as new columns and the new cells are populated with NaN value.

The ```concat()``` function appends the rows from the two Dataframes to create the df_all_rows Dataframe. When you list this out you can see that all of the data rows are there, however, there is a problem with the index (It is restarting with 0)
"""

## concat s_a and s_b
df1 = pd.DataFrame({'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'Name': ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj'],
       'salary': [1407, 1416, 1272, 1803, 1725, 1280, 1651, 1374, 1628, 1512], 'bonus': [158, 192, 113, 105, 117, 151, 103, 139, 160, 108]})
df2 = pd.DataFrame({'ID': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'Name': ['aa1', 'bb1', 'cc1', 'dd1', 'ee1', 'ff1', 'gg1', 'hh1', 'ii1', 'jj1'],
       'salary': [1607, 1646, 1262, 1063, 1765, 1260, 1661, 1764, 1668, 1562], 'bonus': [128, 122, 123, 205, 127, 121, 123, 129, 102, 182]})

pd.concat([df1, df2], ignore_index = True)

"""We didn’t explicitly set an index for any of the Dataframes we have used. For s_a and s_b default indexes would have been created by pandas. When we concatenated the Dataframes the indexes were also concatenated resulting in duplicate entries.

This is really only a problem if you need to access a row by its index. We can fix the problem with the following code.

What if the columns in the Dataframes are not the same?
"""

df1 = pd.DataFrame({'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'Name': ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj'],
       'sal': [1407, 1416, 1272, 1803, 1725, 1280, 1651, 1374, 1628, 1512], 'bonus': [158, 192, 113, 105, 117, 151, 103, 139, 160, 108]})
df2 = pd.DataFrame({'ID': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 'Name': ['aa1', 'bb1', 'cc1', 'dd1', 'ee1', 'ff1', 'gg1', 'hh1', 'ii1', 'jj1'],
       'salary': [1607, 1646, 1262, 1063, 1765, 1260, 1661, 1764, 1668, 1562], 'bonus': [128, 122, 123, 205, 127, 121, 123, 129, 102, 182]})
#print(df1)
print(pd.concat([df1, df2], join='inner', ignore_index = True))
#see outer join as well

pd.concat([df1, df2], axis = 1)

"""            **    Performance: Which is faster pandas concat or append?**
Well, both are almost equally faster.

However there will be a slight change depending on the data.

1. Append function will add rows of second data frame to first dataframe iteratively one by one. Concat function will do a single operation to finish the job, which makes it faster than append().

2. As append will add rows one by one, if the dataframe is significantly very small, then append operation is fine as only a few appends will be done for the number of rows in second dataframe.

3. Append function will create a new resultant dataframe instead of modifying the existing one. Due to this buffering and creating process, Append operation’s performance is less than concat() function. However Append() is fine if the number of append operation is a very few. If there are a multiple append operations needed, it is better to use concat().
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# import pandas as pd
# df = pd.DataFrame(columns=['A'])
# for i in range(50):
#     df = df.append({'A': i*2}, ignore_index=True)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# df = pd.concat([pd.DataFrame([i*2], columns=['A']) for i in range(50)], ignore_index=True)

"""### Adding the columns from one Dataframe to those of another Dataframe"""

df1 = pd.DataFrame({'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'Name': ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj'],
       'sal': [1407, 1416, 1272, 1803, 1725, 1280, 1651, 1374, 1628, 1512], 'bonus': [158, 192, 113, 105, 117, 151, 103, 139, 160, 108]})
df2 = pd.DataFrame({'dep_id': [17, 27, 37, 47, 57, 67, 77, 87, 97, 107], 'pin': ['1y58', '1u92', 'u113', '1y05', '1tt17', 'y151', '1r03', 'p139', '1o60', '10y8']})

df_all_cols = pd.concat([df1, df2], axis = 1)
df_all_cols

"""We use the ```axis=1``` parameter to indicate that it is the columns that need to be joined together. Notice that the Id column appears twice, because it was a column in each dataset. This is not particularly desirable, but also not necessarily a problem. However, there are better ways of combining columns from two Dataframes which avoid this problem.

## Indexing in Pandas

### Indexing a Dataframe using indexing operator []
"""

dataf = pd.DataFrame({'Name': ['qwe', 'wer', 'ert', 'rty', 'tyu', 'yui', 'uio', 'iop', 'asd', 'sdf', 'dfg', 'fgh', 'ghj', 'hjk', 'jkl', 'zxc', 'xcv', 'cvb', 'vbn', 'bm'],
                      'Age': [51, 64, 19, 10, 79, 53, 65, 14, 34, 46, 56, 78, 26, 17, 34, 51, 27, 46, 70, 56]})
print(dataf)

dataf['Age']

"""### Indexing a DataFrame using .loc[ ] """

# import pandas as pd
# #put here url where nba.csv is uploaded on drive
url = "https://drive.google.com/file/d/1i8FEIBy8SlaRGyA7w5pSauSDS-_zK6M8/view?usp=sharing"
url='https://drive.google.com/uc?id=' + url.split('/')[-2]  

# # making data frame from csv file
nba = pd.read_csv(url, index_col ="Name")


nba.head()

data1 = data.reset_index() #good to know

# retrieving Team columns by loc method
# to see col
nba['Team']

#to see row index 5
nba.loc[5]

# Retrieving few rows and columns by loc method
# Retrieve only Jae Crowder, Jonas Jerebko, John Holland rows
# Print only "Team", "Number", "Salary","College" columns
print(nba.loc[[1,5,4,8,6,7],['Team', 'Number', 'Salary', 'College']])

nba = nba.set_index('Name')

# Retrieving few rows and ALL columns by loc method
# Retrieve only Jae Crowder, Jonas Jerebko, John Holland rows
print(nba.loc[['Jae Crowder', 'Jonas Jerebko', 'John Holland']])

# retrieving ALL rows and few columns by loc method
# Print only "Team", "Number", "Salary","College" columns
print(nba.loc[:,['Team', 'Number', 'Salary', 'College']])

# retrieving rows where salary is greater than 1000000
# Print only "Team", "Number", "Salary","College" columns
selected_data = data.loc[data.Salary>1000000, ["Team", "Number", "Salary","College"]]
  
print(selected_data)

# retrieving rows where salary is greater than 1000000 and college is Texas, Georgia Tech 
# Print only "Team", "Number", "Salary","College" columns
selected_data = data.loc[(data.Salary>1000000) & (data.College=='Texas'), ["Team", "Number", "Salary","College"]]
  
print(selected_data)

"""### Extracting rows using Pandas .iloc[]"""

# retrieving multiple rows by iloc method 
# Retrieve 1, 3, 6 index
multiple_rows = data.iloc[[1,3,6],:] 
  

print(multiple_rows)