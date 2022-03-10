'''
plot the analysis of result
'''

import matplotlib.pyplot as plt
import pandas as pd
from seaborn.rcmod import set_style

def avg_plot(df_,fig_name):
    '''
    plot avg of columns in data and save to given name
    :param df:pd.DataFrame
    :return None
    '''

    assert isinstance(fig_name,str) and isinstance(df_,pd.DataFrame)
    # values of first 9 feats
    f, ax = plt.subplots(figsize=(25, 15))
    for i in range(df_.shape[1]):
        sns.barplot(data=df_.iloc[:, :9])

    plt.ylabel('Scoress', size=20)
    plt.xlabel('Features', size=20)
    plt.xticks(rotation=90, size=20)
    plt.yticks(rotation=0, size=20)
    plt.title('Average scores of features across all cities in the data', size=20)

    f.savefig(fig_name+'1.jpg')

    # values of last 9 feats
    f, ax = plt.subplots(figsize=(25, 15))
    for i in range(df_.shape[1]):
        sns.barplot(data=df_.iloc[:, :9])

    plt.ylabel('Scoress', size=20)
    plt.xlabel('Features', size=20)
    plt.xticks(rotation=90, size=20)
    plt.yticks(rotation=0, size=20)
    plt.title('Average scores of features across all cities in the data', size=20)

    f.savefig(fig_name + '2.jpg')

def rankPlots(df, category):
    '''
    plot finale rank for each category
    :param: df: category ranks dataframe
            category: category name
    :return: None
    '''
    assert isinstance(df,pd.DataFrame) and isinstance(category,str)

    filePath = 'plots/rank/'+category+'.png'
    title = 'Top 25 US Cities in Preference: '+category
    df_score = pd.DataFrame(zip(df[category], df['UA_Name']))
    df_score.columns = ['Scores', 'City']
    new_index = (df_score['Scores']).index.values
    df_score = df_score.reindex(new_index)
    plt.figure(figsize=(25,15))
    sns.barplot(x='City',y='Scores',data=df_score[:10], palette=sns.cubehelix_palette(15, reverse=False))
    plt.title(title)
    plt.xticks(rotation=90, fontsize= 30)
    plt.savefig(filePath, bbox_inches = 'tight')
