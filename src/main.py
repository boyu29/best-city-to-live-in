'''
main file
'''

from data_cleaning import data_cleaning
from set_category import find_correlation
from assign_weight import assign_weight, cluster_weight
import os
import sys

if __name__=='__main__':
    # read in .csv
    file = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    filename = file + '/data/uaScoresDataFrame.csv'

    # data cleaning
    df,df2=data_cleaning(filename)

    # find colleration between columns
    category,df4=find_correlation(df2)

    #assign weight to 2 parallel cases
    df8,df9=assign_weight(df,df4,category)


    print(df8)
    print(df9)

    print('Process Finished... See plots in plots folder')





