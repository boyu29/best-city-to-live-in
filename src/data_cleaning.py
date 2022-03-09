'''
clean and preprocess data
'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def data_cleaning(filename:str):
    '''
    :param filename: str
    :return:
        df: pd.DataFrame original data
        df2: pd.DataFrame after one-hot-key
    '''

    assert isinstance(filename,str)

    df = pd.read_csv(filename)

    # check for any incompletes or duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # one hot encoding: categorizing the country and continent into numerical values
    # ADD CITY NAME IF YOU WAnt, IDK IF YOU WANT TO USE IT
    # You can also drop if not using.
    df2 = pd.get_dummies(data=df, columns=['UA_Country', 'UA_Continent'])

    print(df2.head())
    print(df.dtypes)

    cols_to_scale = ['Housing', 'Cost of Living', 'Startups', 'Venture Capital', 'Travel Connectivity', 'Commute',
                     'Business Freedom', 'Safety', 'Healthcare', 'Education', 'Environmental Quality', 'Economy',
                     'Taxation', 'Internet Access', 'Leisure & Culture', 'Tolerance', 'Outdoors']
    scaler = MinMaxScaler()
    df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

    return df,df2


