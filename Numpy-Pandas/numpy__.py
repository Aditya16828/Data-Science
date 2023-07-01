import numpy as np

"""homogeneous datatype"""

#autoconvert to string
l1 = [ 56, 12.4556]
print(l1)
na1 = np.array(l1)
na1

"""Syntax of an array in numpy as np.array (object, dtype=None, copy=True, order='K', subok=False, ndmin=0)

```
Object:
Any object, which exposes an array interface whose __array__ method returns any nested sequence or an array.

dtype: (Optional)
This parameter is used to define the desired parameter for the array element. If we do not define the data type, then it will determine the type as the minimum type which will require to hold the object in the sequence. This parameter is used only for upcasting the array.
```
"""

# List capable of contain multiple data types but array can not contain multiple data type every element should be of single data type (homogenous data).

list_0 = [1, 1.0]                     ## List of Multiple Datatypes
list_1 = ['jay', 2, 1.9, 25, list_0]  ## List containing many datatypes
#print(list_1)


# np.array (object, dtype=None)
## Create NumPy array from list [1,2,3,4] of datatype int
array_1 = np.array([1,2,3,4.4], dtype='i')
print(array_1)
## Create NumPy array from list [1,2,3,4] of datatype float
array_2 = np.array([1,2,3,4.4], dtype='f')
print(array_2)
## Create NumPy array from list [1,2,3,4] of datatype str
array_3 = np.array([1,2,3,4.4], dtype='U')
print(array_3)

"""## List vs Array

A list cannot do mathematical operations directly, whereas an array can.
One of the key distinctions between an array and a list is this. A list can store a float or an integer, but it can't really perform mathematical operations on them.
"""

list_1 = [0,1,2]
list_1 * 3

list_1 = [0,1,2]
(list_1)/3

"""Of course, it's possible to do a mathematical operation with a list, but it's much less efficient.

* Arrays need to be declared. Lists don't.
* Arrays can store data very compactly and are more efficient for storing large amounts of data.
"""

arr = np.array([0,1,2])
arr * 2

"""See how using a * (multiply) in a list returns a repeated data in the list (while we meant to multiply all of the data in the list) and where using it on an array gives a correct or desired result.

This is why you should use an array if your data requires a lot of mathematical calculations. However, you may easily add a mathematical function to your list using a numpy function.

But, you still have to use the numpy libraries anyway, so why don’t just use an array?
"""

arr1 = np.array([1,3,5,7,9])
print(type(arr1))
l1 = list(arr1)
print(type(l1))

list_1
print(np.multiply(list_1,2))

print(list(np.multiply(list_1,2)))
#visually arrays dont have a ,

"""### Advantages of using Numpy Arrays Over Python Lists:

* consumes less memory.
* fast as compared to the python List.
* convenient to use.

## Array creation in Numpy

### Creating arrays from multiple methods
"""

# Creating array from list of list [[10, 34, 35], [125, 851, 0]] with type float
float_array = np.array([[10, 34, 35], [125, 851, 0]], dtype='f')
print ("Float numbers array created from list", float_array)
 
# Creating array from tuple (5, 6, 9)
array_from_tuple = np.array((5, 6, 9))
print ("\nArray created from tuple:\n", array_from_tuple)

print(float_array)

print(*range(5,11,2))

# Array creation with arange, np.ones 

# Syntax. np.arrange (end value, dtype=data type)
# Create array from 0 to 4 
arrange_array= np.arange(0,5,1) #arranging array by giving end value
print("arranging array from 0 to 4:\n",arrange_array)

# Syntax. np.arrange (start vaue, end value, interval, dtype=data type)
# Create array from 0 to 6 with interval of 2
arrange_array1 = np.arange(0,7,2)
print("arranging array with given interval:\n",arrange_array1)

# Syntax. np.arrange (end value with data type)
# Create array till 5.0
#arrange_array2=  np.arange(6, dtype='U')
#print("arranging array with given end point:\n",arrange_array2)

#way around
arrange_array2=  np.array(range(6), dtype='U')
print("arranging array with given end point:\n",arrange_array2)

# Syntax. np.ones (size of array, dtype=data type)
## Create an array (2 x 3) of all values as 1
array_with_ones= np.ones(5, dtype='i')
print("arranging array with all ones given rows and columns:\n",array_with_ones)

# Creating a 2X3 array intialised with all zeroes
## Create an array (2 x 3) of all values as 0
array_with_zeros = np.zeros(4, dtype = 'i')
print ("\n Array of defined size initialized with zeros:\n", array_with_zeros)

# Create a 3X2 array intialised with all zeroes
array_with_zeros = np.zeros((3,2))
print ("\n Array of defined size initialized with zeros:\n", array_with_zeros)

one1 = np.ones((5,4,3))
print(one1)

# Create an array with random values of size 3x3
Random_array = np.random.random((3,3))
print ("\n Random numbers array:\n", Random_array)

print(np.random.randint(10,25, size = (5,6)))

"""Rerun the above cell to see random values everytime."""

# List of Tuples to array

l_of_t = [(1, 2, 3), (4, 5, 6)] #Define the tuples

# Syntax. np.array (defined list of tuples)
# Create an array from l_of_t
a = np.array(l_of_t)
print(a)

l_of_l = [[1, 2, 3], [4, 5, 6]] #Define the tuples

a = np.array(l_of_l)
print(a)

"""### linspace method"""

#Create a sequence of 8 values in range 20 to 50
new_array = np.linspace(10, 20, 5)
print ("\nA sequential array with 8 values between"
                                        "20 and 50:\n", new_array)
print(np.linspace(1,10,4))

"""## NumPy Array dimensions"""

np.array(     [     1,2,3,4     ]    )

qw = np.array(     [[[[     [1,2,3,4], [5,6,7,8]  ]]]]    )
qw.ndim

#creating a new array
array_0 = np.array([
                    [[1,2,3,4],[5,6,7,8], [9, 10, 11, 12]],
                    [[10,20,30,40],[50,60,70,80], [90, 100, 101, 102]]], dtype = 'f')
print(array_0)
#syntax.arrayname.shape
array_0.shape
#Each array has attributes shape (the size of each dimension)

"""In (4,) 4 indicates that size of the array and (4,) repesents the dimension of array i.e., (4,) means 1 Dimensional array """

# Creating a new array
array_1 = np.array([[1,2,3,4],[5,4,6,7],[2,3,4,6]])
print(array_1)
# Syntax.arrayname.shape
array_1.shape  
  # shape (the size of each dimension)

"""In (3,4), 3 indicates that number of rows and 4 indicates number of column. and (3,4) indicates it is a 2 dimensional array"""

# understanding a 3D array
array_2 = np.array([[[1,2,3,4],[5,4,6,7],[2,3,4,6]],
                    [[3,4,5,6],[5,6,7,8],[9,10,11,12]]])
print(array_2)

# Shape of an array
array_2.shape 
#syntax.arrayname.shape

"""In (1, 3, 4), 1 indicates the added dimension to array, remaining 3 and 4 are same as above, (1,3,4) indicates it is a 3 dimensional array"""

# to find the data type of an array
# syntax arrayname.dtype
print("data type of array_0:\n", array_0.dtype)  
print("data type of arrange_array:\n", arrange_array.dtype) 
print("data type of array_2:\n", array_2.dtype)

# To know the dimension of an array(1D,2D,3D etc..)
# syntax.arrayname.ndim
print("the dimension of array_0:\n", array_0.ndim)
print("the dimension of array_0:\n", array_1.ndim)
print("the dimension of array_0:\n", array_2.ndim)

"""### Reshaping array"""

# 1D to 2D and vice-versa
# Reshaping 1X6 array to 3X2 array

# orig_arr_1 = np.array([10, 32, 36, 34, 4, 55]) ## 1D array

# reshaped_arr_1 = orig_arr_1.reshape(3, 2)
 
# print ("\nOriginal array:\n", orig_arr_1)
# print ("\n Reshaped array:\n", reshaped_arr_1)
# print ("\n Dimension of original array:\n", orig_arr_1.ndim)
# print ("\n dimension of Reshaped array:\n", reshaped_arr_1.ndim)

orig_arr_2 = np.array([[10, 32, 36], [34, 4, 55], [4,77,67]])   ## 2D Array
 
reshaped_arr_2 = orig_arr_2.reshape(9)
print ("\nOriginal array:\n", orig_arr_2)
print ("\n Reshaped array:\n", reshaped_arr_2)
print ("\n dimension of original array:\n", orig_arr_2.ndim)
print ("\n dimension of Reshaped array:\n", reshaped_arr_2.ndim)

# 1D to 3D and vice-versa
# Reshaping 1X4 array to 1X2X2 array
orig_arr_3 = np.array([10, 32, 36, 4])
 
reshaped_arr_3 = orig_arr_3.reshape(1 ,2, 2)

 
print ("\nOriginal array:\n", orig_arr_3)
print ("\n Reshaped array:\n", reshaped_arr_3)
print ("\n dimension of original array:\n", orig_arr_3.ndim)
print ("\n dimension of Reshaped array:\n", reshaped_arr_3.ndim)

orig_arr_4 = np.array([[[10, 32, 36], [34, 4, 55], [4,5,6]]])
 
reshaped_arr_4 = orig_arr_4.reshape(9,)  ## Reshaping is not possible
print ("\nOriginal array:\n", orig_arr_4)
print ("\n Reshaped array:\n", reshaped_arr_4)
print ("\n dimension of original array:\n", orig_arr_4.ndim)
print ("\n dimension of Reshaped array:\n", reshaped_arr_4.ndim)

# Reshaping 2X4 array to 2X2X2 array
orig_arr = np.array([[10, 32, 36, 34],
                [56, 19, 75, 20]])
 
reshaped_arr = orig_arr.reshape(2, 2, 2)
 
print ("\nOriginal array:\n", orig_arr)
print ("\n Reshaped array:\n", reshaped_arr)

"""### Flatten array"""

# Flatten array
arr = np.array([[[1, 2, 3], [4, 5, 6]]])
flarr = arr.flatten()

print ("\nOriginal array:\n", arr)
print ("Fattened array:\n", flarr)

"""## Array Indexing

### Slicing
"""

array_1D = np.array([10, 32, 36, 34]) #1 Dimensonal array indexing
array_1D[1] # calling 32

array_1D[-1]   # calling 34 with negative index

#multidimentional array indexing
array_multiD = np.array([
                [100, 14, 76, 54],
                [125, 81, 0, -24],
                [566, 159, 175, 20],
                [38, -47, 84, 22.0]],dtype = int)

print("Shape of the array is: ", array_multiD.ndim, array_multiD.shape)

## 81 is in the 1st row and 1st column
print("Accessing 1st row: ", array_multiD[0])
print("Accessing 1st column: ", array_multiD[:, 1])  ## : indicates all the rows
print("Accessing (1st row, 1st column): ", array_multiD[2,1])  # calling 159

array_multiD[3,2] # calling 84

array_multiD[0,3] # calling 54

## Accesing 4th row, 3rd column
array_multiD[4,3]

# An exemplar array
array_example = np.array([[10, 32, 36, 34],
                [125, 851, 0, -24],
                [56, 19, 75, 20],
                [3, -7, 4, 2.0]])

# Slicing array
## Access first 2 rows
sliced_array = array_example[0:2, :]   #:2 returns the 0,1 indexed row
print(sliced_array)

## Access first 2 columns
sliced_array = array_example[:, :2]   #:2 (after comma) returns the 0,1 indexed column
print(sliced_array)

# Slicing array
sliced_array = array_example[:2, :2]   #:2 returns the 0,1 indexed row and ::2 returns the 0 and 2 nd column
print ("Array with first 2 rows and alternate"
                    "columns(0 and 2):\n", sliced_array)

## Accesing the 3rd row
array_example[3:] ## When accessing only row(s). Mentioning anything after : is optional. It will take all by default.

# Slicing array
## Accessing 1st, 2nd row and 2nd, 3rd column
sliced_array = array_example[1:3,2:4]
print ("1st&2nd row, 2&3rd column\n", sliced_array)

"""### Integer array indexing"""

# An exemplar array
array_example = np.array([[10, 32, 36, 34],
                [125, 851, 0, -24],
                [56, 19, 75, 20],
                [3, -7, 4, 2.0]])

# Integer array indexing example
Integer_index_array = array_example[[0, 1, 2, 3], [3, 2, 1, 0]]
print ("\nElements at indices (0, 3), (1, 2), (2, 1),"
                                    "(3, 0):\n", Integer_index_array)

arr_3d = np.array([[[1,2,3,4],[5,4,6,7],[2,3,4,6]],
                   [[3,4,5,6],[5,6,7,8],[9,10,11,12]]])
print(arr_3d)
# to get 7+9 = 16

arr_3d[0,1,3] + arr_3d[1][2][0]

"""### Boolean array indexing"""

# An exemplar array
array_example = np.array([[10, 32, 36, 34],
                [125, 851, 0, -24],
                [125, 19, 75, 20],
                [3, -7, 4, 2.0]])

# boolean array indexing example
# condition = array_example > 40 # cond is a boolean array
# boolean_output = array_example[condition]
# print ("\nElements greater than 40:\n", boolean_output)

# OR

arr2 = array_example[array_example>=12]
arr2

"""## Manipulating Arrays"""

#Items of an array can be modified in any place of an array like below
array = np.ones((4, 5))
print("original array\n",array)

## Replace a value at 2nd row, 2nd column to 10
array[2, 2] = 10
print("modified array\n",array)

## Replace a value at 3rd row to 11
array[3] = 11

print(array)

## Replace a value at 1st column to -12
array[:, 1] = -12

print(array)

"""## Basic arithmetic operations

### Operations on a single array
"""

sample_arr = np.array([[5, 215, 69],
                       [44, 74, 21],
                       [30, 12, 94]])
 
# maximum element of array
print ("Largest element of array :", sample_arr.max())
print ("Row-wise maximum elements:",
                    sample_arr.max(axis = 1))
 
# minimum element of array
print ("Minimum element of array :", sample_arr.min())
print ("Column-wise minimum elements:",
                        sample_arr.min(axis = 0))
 
# sum of array elements
print ("Sum of all array elements:",
                            sample_arr.sum())
 
# cumulative sum along each row
print ("Cumulative sum along each row:\n",
                        sample_arr.cumsum(axis = 1))
 
# cumulative sum along each column
print ("Cumulative sum along each column:\n",
                        sample_arr.cumsum(axis = 0))

#col sum
print ("Sum of all array elements:",
                            sample_arr.sum(axis=0))
#row sum
print ("Sum of all array elements:",
                            sample_arr.sum(axis=1))

"""### Binary operators"""

arr_a = np.array([[5, 7],
                  [8, 1]])
arr_b = np.array([[4, 1],
                  [9, 2]])
 
# addition of arrays
print ("Array sum:\n", arr_a + arr_b)
print(arr_a + 1000)
# multiplication of arrays (elementwise multiplication)
print ("Array multiplication:\n", arr_a*arr_b)
 
# matrix multiplication of arrays
print ("Matrix multiplication:\n", arr_a.dot(arr_b))

print(arr_a @ arr_b)

"""## Sort Method"""

sample_arr = np.array([[1, 4, 2],
                       [3, 4, 6],
                       [0, -1, 5]])
 
# sorted array 
print ("Array elements in sorted order:\n",
                    np.sort(sample_arr, axis = None))
# sorted array col wise
print ("Array elements in sorted order:\n",
                    np.sort(sample_arr, axis = 0))
 
# sort array row-wise
print ("Row-wise sorted array:\n",
                np.sort(sample_arr, axis = 1))

# Example to show sorting of structured array
# set alias names for dtypes
dtypes = [('name', 'U10'), ('graduation_year', int), ('Percentage', float)]
 
# Values to be put in array
values = [('Kartik', 2001, 96.5), ('Pranay', 2001, 85.7),
           ('Arjav', 1999, 76.9), ('Krunal', 2003, 90.0)]
            
# Creating array
created_arr = np.array(values, dtype = dtypes)
print ("\nArray sorted by names:\n",
            np.sort(created_arr, order = 'name'))

"""## Data Type"""

#create array with data type integer
arr = np.array([3, 6, 0, -1], dtype='i')
print(arr.dtype)

#coverting datatype of existing array
converted_arr = arr.astype('bool')
print(arr)
print(converted_arr.dtype)
print(converted_arr)

"""## numpy array join"""

arr_concat1 = np.array([2, 2, 6, 8])
arr_concat2 = np.array([1, 3, 5, 7])
final_arr = np.concatenate((arr_concat1, arr_concat2))
print("1d concatenated array : \n",final_arr)


#joining a 2D array
arr2d_concat1 = np.array([[3, 5],
                          [7, 9]])
arr2d_concat2 = np.array([[0, 6], 
                          [2, 1]])

arr_concat_final = np.concatenate((arr2d_concat1, arr2d_concat2), axis = 0)
print("\n 2d concatenated array: \n",arr_concat_final)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

print(np.concatenate((a, b.T), axis=1))
print(np.concatenate((a, b), axis=None))

print(b.T)
print(a)

"""## numpy array search"""

# find the indexes where values are odd
arr_all = np.array([123, 345, 29, 24, 80, 61, 79, 84])

arr_odd_index = np.where(arr_all%2 == 1)

print(arr_odd_index)

"""## numpy array split

### Split Array function
"""

array_to_split = np.array([6, 1, 7, 9, 10, 4, 43, 28, 54])

split_array = np.array_split(array_to_split , 3)

print(split_array)

print(split_array[0])
print(split_array[1])
print(split_array[2])
#easier than reshape

array_to_split.reshape((3,3))
# this is a single array
array_to_split.reshape((3,3))[0]

"""### Horizontal Split"""

array = np.arange(36.0).reshape(6, 6)
  
hsplit_arr = np.hsplit(array, 3)
  
print("Array before splitting : \n", array)
print("\n Array after horizontal split : \n \n",hsplit_arr)

print(type(np.hsplit(array, 3)))
print(len(np.hsplit(array, 3)))

print(type(hsplit_arr))
hsplit_arr[0]

"""### Vertical Split"""

array = np.arange(36.0).reshape(6, 6)
  
vsplit_arr = np.vsplit(array, 3)
  
print("Array before splitting : \n", array)
print("\n Array after vertical split : \n \n",vsplit_arr)

print(type(np.vsplit(array, 3)))
print(len(np.vsplit(array, 3)))

type(vsplit_arr)

print(hsplit_arr[0])

"""## Numpy Stacking"""

arr_1 = np.array([[7, 3],
                 [1, 4]])
  
arr_2 = np.array([[1, 9],
                  [4, 0]])

"""### Vertical Stacking


"""

# vertical stacking
print("Vertical stacking:\n", np.vstack((arr_1, arr_2)))

"""### Horizontal Stacking"""

# horizontal stacking
print("\nHorizontal stacking:\n", np.hstack((arr_1, arr_2)))

"""### Column Stacking"""

column = [5, 6]
  
# stacking columns
print("\nColumn stacking:\n", np.column_stack((arr_1, column)))

# row stack is vstack
import numpy as np
arr_1 = np.array([[1, 2],
                 [3, 4]])
  
arr_2 = np.array([[5, 6],
                  [7, 8]]) 

arr_3 = np.array([[9, 10],
                  [11, 12]])

stacked = np.vstack((arr_1, arr_2, arr_3))
print("Vertical stacking:\n", np.vstack((arr_1, arr_2, arr_3)))
print('vertical split: \n', np.vsplit(stacked, 3))

import numpy as np
arr_1 = np.array([[1, 2],
                 [3, 4]])
  
arr_2 = np.array([[5, 6],
                  [7, 8]]) 

arr_3 = np.array([[9, 10],
                  [11, 12]])

stacked = np.hstack((arr_1, arr_2, arr_3))
print("Vertical stacking:\n", np.hstack((arr_1, arr_2, arr_3)))
print('vertical split: \n', np.hsplit(stacked, 3))

"""## Broadcasting

Broadcasting is the ability of a of NumPy to treat arrays of different shapes during arithmetic operations.
If two arrays are of exactly the same shape, then these operations are smoothly performed.
"""

a = np.array([1,2,3,4]) 
b = np.array([1,80,33,44]) 
c = a * b 
print("multiplication of the two arrays",c)

c*10

"""If two or any set of arrays brodcastable then they are atleast follow any one of the following rules.
1. Arrays have exactly the same shape.
2. Arrays have the same number of dimensions and the length of each dimension is either a common length or 1.
3. Arrays having too few dimensions can have their shape prepended with a dimension of length 1 so that the above-stated property is true.
"""

# Rule 1 - Passed
a = np.array([5, 11, 25,54])
b = np.array([5, 5, 5, 63])
c = a + b
print("adding two arrays",c)

# Rule 1 - Failed
# Rule 2 - Failed
a = np.array([5, 11, 25,54])
b = np.array([5, 5, 5])
c = a + b
print("adding two arrays",c)

# Rule 1 - Failed
# Rule 2 - Pass
a + 5  ## 5 is scalar value and defult of dimension 1

"""### Scalar operation on an array


In this example, the scalar is stretched to become an array of with the same shape as a so the shapes are compatible for element-by-element multiplication
"""

array = np.array([5, 6, 7])
  
# Example 1
scalar = 2.0
print("output of scalar multiplication : ", scalar * array)
  
# Example 2
array_2 = [2.0, 2.0, 2.0]
print("output on arrays multiplication :",array * array_2)

array_2 - np.ones((1,3))*2

"""We can similarly extend this to arrays of higher dimensions. Observe the result when we add a one-dimensional array to a two-dimensional array:"""

a = np.array([[0, 1, 2], [3,4,5], [6,7,8]])
M = np.ones((3, 3))

print("output of array addition :\n",M+a)

a = np.array([0, 1, 2])
M = np.ones((3, 3))

print(a.reshape((3,1)))
print(M, '\n\n')

print("output of array addition :\n",M+a.reshape((3,1)))

a = np.array([0, 1, 2])
b = np.array([3,4,5])

print(a)
print(b.reshape((3,1)))

print(a + b.reshape((3,1)))

"""### Broadcasting where both arrays gets stretched

In below example, broadcasting stretches both arrays to form an output array larger than either of the initial arrays

np.newaxis is used to increase the dimension of the existing array by one more dimension when used once. Thus,

1D array will become 2D array.2D to 3D
"""

a = np.arange(3)
b = np.arange(2)[:, np.newaxis] #it is equal to b.reshape(2,1) 
c = a+b
print(c)
print(c.shape)

print(c[:,np.newaxis])
# is same as 
c.shape
#print(c.reshape(2,3,2))

print(np.arange(2))
print(np.arange(2)[:, np.newaxis])

import numpy as np
  
a = np.array([5.0, 12.0, 42.0, 75.0])
b = np.array([1.0, 6.0, 8.0])
  
print(a[:, np.newaxis] + b)

