from EDA import univariates, bivariates
from utils.data_import import load_data
from preprocessing.preprocess import preprocess_data
from models.pipelines import run_model_script

def rerunQ():
    if input('Do you want to do something else? (y/n)') == 'n':
        break
    else:
        continue


def explore_script(df):
    step2 = input('What do you want to do? \n 1: Explore univariates\n '
                  '2: Explore bivariates\n 3: Both')
    target = input('What is the target variable?')
    columnsQ = input('Would you like to explore specific columns? (y/n)')
    if columnsQ == 'n':
        columns = df.columns
    if columnsQ == 'y':
        columns = input('What columns?').split(',')
        columns = [column.strip() for column in columns]
    if step2 == '1':
        univariates.explore_df(df, columns)
    elif step2 == '2':
        bivariates.explore_df(df, target, columns)
    elif step2 == '3':
        univariates.explore_df(df, target, columns)
        bivariates.explore_df(df, target, columns)
    rerunQ()

def script():
    source = input('Give filepath')
    filetype = input('Give filetype \n csv \n excel \n JSON \n parquet \n' 
                     ' csv_with_delimiter \n csv_with_semicolon')
    df = load_data(source, filetype)
    loop = True
    while loop:
        step1 = input('What do you want to do? \n 1: Explore raw data'
                      '\n 2: Clean data\n 3: Explore'
                      ' cleaned data\n 4: Run a model')
        if step1 == '1':
            print('exploring raw data')
            explore_script(df)

        elif step1 == '2':
            preprocess_data(df)
            rerunQ()

        elif step1 == '3':
            print('exploring cleaned data')
            explore_script(load_data('./src/preprocessed_data.csv', 'csv'))

        elif step1 == '4':
            step2 = input('What data? \n 1: Clean data\n '
                          '2: Raw data')
            if step2 == '1':
                df = load_data('./src/preprocessed_data.csv', 'csv')
            targetQ = input('Is the target variable categorical? (y/n)')
            if targetQ == 'y':
                modelQ = input('What model type?'
                               '\n 1: Gradient Boosting'
                               '\n 2: Ada Boosting'
                               '\n 3: Random Forest'
                               '\n 4: KNN'
                               '\n 5: Decision Tree'
                               '\n 6: Logistic Regression'
                               )
            elif targetQ == 'n':
                modelQ = input('What model type?'
                               '\n 7: Linear Regression'
                               '\n 8: Ridge'
                               '\n 9: Lasso'
                               '\n 10: ElasticNet'
                               '\n 11: Support Vector'
                               '\n 12: Decision Tree'
                               '\n 13: Random Forest')
            target = input('What is the target variable?')
            columnsQ = input('Would you like to only include specific columns? (y/n)')
            if columnsQ == 'n':
                columns = df.columns
            if columnsQ == 'y':
                columns = input('What columns?').split(',')
                columns = [column.strip() for column in columns]
            run_model_script(df, target, columns, modelQ)
        rerunQ()