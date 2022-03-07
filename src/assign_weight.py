'''
assign weight to categorized data
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from set_category import Category
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, normalize
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os


def plot_invariance_on_continent(l,df,fig_name):
    '''
    plot bar chart on continents for category
    :param l: list of column names to plot
    :return:
    '''
    file = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    filename = file + '/plots/'

    color_list = ['black', 'red', 'yellow', 'green', 'blue', 'purple', 'orange']

    # plot bar on continents for category1
    f, ax = plt.subplots(figsize=(15, 5))
    for i in range(len(l)):
        sns.barplot(y='UA_Continent', x=l[i], data=df, color=color_list[i], alpha=0.5,
                    label=l[i])

    plt.ylabel('Continents')
    plt.xlabel('Values')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.title('Average Values of the Continents')
    ax.legend(loc='upper right', frameon=True)

    f.savefig(filename+fig_name+'.jpg')


def assign_weight(df,df4,category):
    '''
    weight of columns are unbalanced, make columns have the same mean and assign weight
    :param df: pd.DataFrame categorized
    :return: df_new: pd.DataFrame with vlaues 0-10
    '''

    # process data: normalize each column, keep the 'UA_Continent' column
    data_con = df.drop(columns=['Unnamed: 0'])
    column_list = list(df.columns)
    cols_to_scale = column_list[4:]
    scaler = MinMaxScaler()
    data_con[cols_to_scale] = scaler.fit_transform(data_con[cols_to_scale])


    # plots for category1 - 4 on continents
    # check the invariance of data based on continent
    for i in range(4):
        plot_invariance_on_continent(category.cat[i],data_con,'category '+str(i+1))

    for i in range(4):
        plot_invariance_on_continent(category.manual[i], data_con, category.name_mannul[i])

    # check the mean value of each column
    df_temp = data_con[cols_to_scale] * 10
    df_mean = df_temp.mean()
    df_var = df_temp.var()
    print('mean\n', df_mean, '\n')
    print('var\n', df_var)
    ### var does not vary much, we just have to ajust the mean value to the same level.
    ### annother approach would be ajust the mean value according to each continent(haven't tried out yet)

    # Assign weight to 2 parallel case
    ### method
    #   get mean for each column
    #   adjust the mean value of each column to around 5, keep the variance unmodified
    #   vlaue for each category=avg(each column in this category)
    ###
    for item in cols_to_scale:
        df_temp[item] = df_temp[item] - df_mean.loc[item] + 5

    df4['category1'] = (df4['Housing'] + df4['Cost of Living']) / 2
    df4['category2'] = (df4['Travel Connectivity'] + df4['Commute'] + df4['Startups'] + df4['Venture Capital'] + df4[
        'Leisure & Culture'] + df4['Outdoors']) / 6
    df4['category3'] = (df4['Business Freedom'] + df4['Healthcare'] + df4['Education'] + df4['Environmental Quality'] +
                        df4['Economy'] + df4['Internet Access'] + df4['Tolerance']) / 7
    df4['category4'] = (df4['Safety'] + df4['Taxation']) / 2


    df4['Vacation Lovers'] = (df4['Startups'] + df4['Venture Capital'] + df4['Business Freedom'] + df4['Taxation'] +
                              df4['Economy']) / 5
    df4['Entrepreneur & Business person'] = (df4['Travel Connectivity'] + df4['Commute'] + df4['Leisure & Culture'] +
                                             df4['Internet Access']) / 4
    df4['Stability Seeker'] = (df4['Housing'] + df4['Cost of Living'] + df4['Tolerance'] + df4['Outdoors']) / 4
    df4['Family'] = (df4['Safety'] + df4['Healthcare'] + df4['Education'] + df4['Environmental Quality']) / 4

    df8 = df4[['category1', 'category2', 'category3', 'category4']]
    df9 = df4[['Vacation Lovers', 'Entrepreneur & Business person', 'Stability Seeker', 'Family']]
    cols_to_scale = ['category1', 'category2', 'category3', 'category4']
    scaler = MinMaxScaler()
    df8[cols_to_scale] = scaler.fit_transform(df8[cols_to_scale])
    df8 = df8 * 10
    scaler = MinMaxScaler()
    cols_to_scale = ['Vacation Lovers', 'Entrepreneur & Business person', 'Stability Seeker', 'Family']
    df9[cols_to_scale] = scaler.fit_transform(df9[cols_to_scale])
    df9 = df9 * 10

    df8 = df8.round(2)
    df9 = df9.round(2)

    return df8,df9

def cluster_weight(df):
    '''
    Getting Weights (importance) of each category for classifying given city into four categories
    :param: df pd.DataFrame
    :return: weight list
    '''

    file = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    filename = file + '/plots/'

    # first we cluster given data into four clusters using k means clustering
    kmeans = KMeans(
        init="random",
        n_clusters=4,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    df10 = df.drop(columns=['UA_Name', 'Unnamed: 0'])

    le = LabelEncoder()
    df10['UA_Country'] = le.fit_transform(df10['UA_Country'])
    df10['UA_Continent'] = le.fit_transform(df10['UA_Continent'])

    df10['UA_Country'] = (df10['UA_Country']).astype('category')
    df10['UA_Continent'] = (df10['UA_Continent']).astype('category')

    df11 = normalize(df10)

    kmeans.fit(df11)

    # get labels of cities based on above clusters
    y = kmeans.labels_
    X = df11

    data_dmatrix = xgb.DMatrix(data=X, label=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
    xg_reg = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 100)
    xg_reg.fit(X_train, y_train)
    preds = xg_reg.predict(X_test)

    print('accuracy score: ',accuracy_score(preds, y_test))

    #how decision is made with what threshold on classifying given city into one of four categories
    fig1=xgb.plot_tree(xg_reg, num_trees=0)
    fig1.figure.tight_layout()
    fig1.figure.savefig(filename+'tree.png')

    #importance of each feature in classifying each city into one of four categories
    fig2=xgb.plot_importance(xg_reg)
    fig2.figure.tight_layout()
    fig2.figure.savefig(filename+'importance.png')


    print('importance: ',xg_reg.feature_importances_)

    return xg_reg.feature_importances_





