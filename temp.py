import pandas as pd
from pycaret import regression as reg
from pycaret import classification as cla


def reg_compare(df, target):
    reg.setup(data = df, target = target, session_id=123, silent=True, verbose=False)
    result_df = reg.compare_models(verbose=False)
    # result_df = pull()
    return result_df


def cla_compare(df, target):
    cla.setup(data = df, target = target, session_id=123, silent=True, verbose=False)
    result_df = cla.compare_models(verbose=False)
    # result_df = pull()
    return result_df


def clean(df):
    df = df[~df[' Income '].isnull()]
    df[' Income '] = df[' Income '].apply(lambda x: x[1:].replace(',', '').split('.')[0])
    df[' Income '] = df[' Income '].astype('int32')

    df['Education'] = df['Education'].astype('category')
    df['Marital_status'] = df['Marital_Status'].astype('category')
    df['Country'] = df['Country'].astype('category')

    df.drop(['ID', 'Dt_Customer'], inplace=True, axis=1)
    return df


def main():
    df = pd.read_csv("./dataset/regression/marketing_data.csv")

    df = clean(df)

    best_model = reg_compare(df, ' Income ')
    result_df = reg.pull()
    # print(best_model)
    # print(result_df)


if __name__ == "__main__":
    main()