import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def prepare_data(df, train_size=None, shuffle=True, random_state=None):
    # Drop unnecessary columns
    df.drop(columns=['Month', 'Browser', 'OperatingSystems'], inplace=True)

    # Convert boolean columns to integer
    cols = df.columns[df.dtypes == 'bool']
    df[cols] = df[cols].astype(int)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])

    # Split data into features and target
    X = df.drop(columns='Revenue')
    y = df['Revenue']
    return train_test_split(X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)


if __name__ == "__main__":
    df = pd.read_csv('data/project2_dataset.csv')
    X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.7, random_state=42)

    min_max = MinMaxScaler()
    # I saw on the paper that the ranges of the columns are drastically different
    # so I decided to scale them
    scaler_map = {
        "Administrative": MinMaxScaler(),
        "Administrative_Duration": MinMaxScaler(),
        "Informational": MinMaxScaler(),
        "Informational_Duration": MinMaxScaler(),
        "ProductRelated": MinMaxScaler(),
        "ProductRelated_Duration": MinMaxScaler(),
        "BounceRates": MinMaxScaler(),
        "ExitRates": MinMaxScaler(),
        "PageValues": MinMaxScaler(),
        "SpecialDay": MinMaxScaler(),
    }

    # Scale the columns
    for column in scaler_map:
        scaler = scaler_map[column]
        X_train[column] = scaler.fit_transform(X_train[[column]])
        X_test[column] = scaler.transform(X_test[[column]])

    MAX_ITER = 1e+6
    model = LogisticRegression(max_iter=int(MAX_ITER))
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))
    # rows are true labels, columns are predicted labels
    print(confusion_matrix(y_test, model.predict(X_test)))