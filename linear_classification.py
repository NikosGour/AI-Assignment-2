import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def prepare_data(df, train_size=None, shuffle=True, random_state=None):
    df.drop(columns=['Month', 'Browser', 'OperatingSystems'], inplace=True)
    cols = df.columns[df.dtypes == 'bool']
    df[cols] = df[cols].astype(int)
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])
    X = df.drop(columns='Revenue')
    y = df['Revenue']
    return train_test_split(X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)


if __name__ == "__main__":
    df = pd.read_csv('data/project2_dataset.csv')
    X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.7, random_state=42)

    min_max = MinMaxScaler()
    scaling_columns = X_train.columns[(X_train.dtypes == 'int64') | (X_train.dtypes == 'float64')]
    min_max = min_max.fit(X_train[scaling_columns])

    X_train[scaling_columns] = min_max.transform(X_train[scaling_columns])
    X_test[scaling_columns] = min_max.transform(X_test[scaling_columns])

    MAX_ITER = 1e+6
    model = LogisticRegression(max_iter=int(MAX_ITER))
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    print(model.coef_)