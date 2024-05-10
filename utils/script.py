from EDA import univariates, bivariates
from utils.data_import import load_data
from preprocessing.preprocess import preprocess

def rerunQ():
    if input('Do you want to do something else? (y/n)') == 'n':
        break
    else:
        continue


def explore_script(df):
    step2 = input('What do you want to do? \n 1: Explore univariates\n '
                  '2: Explore bivariates\n 3: Both')
    columnsQ = input('Would you like to explore specific columns? (y/n)')
    if columnsQ == 'n':
        columns = None
    if columnsQ == 'y':
        columns = input('What columns?').split(',')
    if step2 == '1':
        univariates.explore_df(df, columns)
    elif step2 == '2':
        bivariates.explore_df(df, columns)
    elif step2 == '3':
        univariates.explore_df(df, columns)
        bivariates.explore_df(df, columns)
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
            preprocess(df)
            rerunQ()
        elif step1 == '3':
            explore_script(load_data('./src/preprocessed_data.csv', 'csv'))

        elif step1 == '4':
            step2 = input('What data? \n 1: Clean data\n '
                          '2: Raw data')
            if step2 == '1':
                df = load_data('./src/preprocessed_data.csv', 'csv')
            modelQ = input('What model type? \n 1: Clean data\n '
                           '2: Raw data')
            if columnsQ == 
            saveQ = input('Do you want to save the model metrics? (y/n)')
        rerunQ()