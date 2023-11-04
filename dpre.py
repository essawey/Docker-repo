def dpre(df):
    from load import load_dataset
    import pandas as pd
    import numpy as np

    df = load_dataset("titanic.csv")


    # Data Transformation
    
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df["Title"] = df["Name"].str.extract(r', (Mr|Miss|Mrs)\.')

    # Data Cleaning

    mean_age = df['Age'].mean()
    std_age = df['Age'].std()
    null_count = df['Age'].isnull().sum() # 263
    null_indices = df[df['Age'].isnull()].index
    missing_ages = np.random.normal(mean_age, std_age, null_count)
    missing_ages = np.clip(missing_ages, 0, df['Age'].max())
    df.loc[null_indices, 'Age'] = missing_ages


    title_counts = df['Title'].value_counts()
    distribution = title_counts / title_counts.sum()
    null_count = df['Title'].isnull().sum() # 95
    null_indices = df[df['Title'].isnull()].index
    missing_titles = np.random.choice(distribution.index, null_count, p=distribution)
    df.loc[null_indices, 'Title'] = missing_titles


    survived_counts = df['Survived'].value_counts()
    distribution = survived_counts / survived_counts.sum()
    null_count = df['Survived'].isnull().sum() # 418
    null_indices = df[df['Survived'].isnull()].index
    missing_survived = np.random.choice(distribution.index, null_count, p=distribution)
    df.loc[null_indices, 'Survived'] = missing_survived


    # Data Reduction
    df.drop('PassengerId', axis=1, inplace=True)

    df.drop('Name', axis=1, inplace=True)

    # Data Discretization

    bins = [df['Age'].min()-1, 10, 20, 30, 40, 50, 60, df['Age'].max()+1]
    labels = ['0-10 years', '11-20 years', '21-30 years', '31-40 years', '41-50 years', '51-60 years', '61+ years']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)


    fare_bins = [df['Fare'].min()-1, 20, 40, 60, 80, 100, df['Fare'].max()-1]
    fare_labels = ['0-20', '21-40', '41-60', '61-80', '81-100', '100+']
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df['FareGroup'] = pd.cut(df['Fare'], bins=fare_bins, labels=fare_labels)

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].values[0])
    df['IntAgeGroup'] = pd.factorize(df['AgeGroup'])[0]
    df['IntFareGroup'] = pd.factorize(df['FareGroup'])[0]
    df['IntTitle'] = pd.factorize(df['Title'])[0]
    df['IntSex'] = pd.factorize(df['Sex'])[0]
    df['IntEmbarked'] = pd.factorize(df['Embarked'])[0]
    df['IntTicket'] = pd.factorize(df['Ticket'])[0]

    
    df.drop('AgeGroup', axis=1, inplace=True)
    df.drop('FareGroup', axis=1, inplace=True)
    df.drop('Title', axis=1, inplace=True)
    df.drop('Sex', axis=1, inplace=True)
    df.drop('Embarked', axis=1, inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    df.drop('Ticket', axis=1, inplace=True)


    df.to_csv("res_dpre.csv", index=False)
    return df