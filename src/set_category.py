import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as spc
import os
import sys

class Category():
    def __init__(self,category_lis,manual_list,name_manual):
        self.cat=category_lis
        self.manual=manual_list
        self.name_mannul=name_manual

def find_correlation(df2):
    '''
    plot correlation matrix
    :param df2: pd.DataFrame
    :return:
        category_list: list assigning 4 categories from correlation matrix
        manual_list: list from manually selected
    '''

    file = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    filename = file + '/plots/'

    df3 = df2[df2.columns[1:19]]

    fig, ax = plt.subplots(figsize=(10, 10))

    hm = sns.heatmap(df3.corr());
    fig = hm.get_figure()
    ax.figure.tight_layout()
    fig.savefig(filename+'heatmap.jpg')

    #remove duplicate values and columns that are correlated with themselves and save it in a file
    c = df3.corr()
    s = c.unstack()
    so = s.sort_values(kind="quicksort")
    filtered_list = so[so >= 0][::-2]
    non_ones = filtered_list[filtered_list != 1.000]
    non_ones.to_csv("correlation_list_reverse.csv")

    df4 = df3.drop(columns=["UA_Name"])

    #create 4 categories using the correlations we have derived

    df5 = df4
    corr = df5.corr().values
    print(df5.columns)
    pdist = spc.distance.pdist(corr)
    linkage = spc.linkage(pdist, method='complete')
    idx = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')
    dict = {}
    for i in range(len(idx)):
        if dict.get(idx[i]):
            dict[idx[i]].append(df5.columns[i])
        else:
            dict[idx[i]] = [df5.columns[i]]
    print(dict)

    # define parameters

    # category_list from correlation heatmap
    category_list = [['Housing', 'Cost of Living'],
                     ['Travel Connectivity', 'Commute', 'Startups', 'Venture Capital', 'Leisure & Culture', 'Outdoors'],
                     ['Business Freedom', 'Healthcare', 'Education', 'Environmental Quality', 'Economy',
                      'Internet Access', 'Tolerance'],
                     ['Safety', 'Taxation']]

    # manual_list from manually selected categories
    manual_list = [['Startups', 'Venture Capital', 'Business Freedom', 'Taxation', 'Economy'],
                   ['Travel Connectivity', 'Commute', 'Leisure & Culture', 'Internet Access'],
                   ['Housing', 'Cost of Living', 'Tolerance', 'Outdoors'],
                   ['Safety', 'Healthcare', 'Education', 'Environmental Quality']]

    # names for manually selected category
    name_manual=['Vacation Lovers','Entrepreneurs & Businesspersons','Stability Seekers','Family']

    category=Category(category_list,manual_list,name_manual)

    return category,df4




