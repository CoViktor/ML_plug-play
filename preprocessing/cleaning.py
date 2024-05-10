def prepare_model_data(cleaned_data_file):
    # Load cleaned data
    cleaned_data = pd.read_csv(cleaned_data_file)

    # Remove outliers
    cleaned_data = remove_outliers(cleaned_data, 'PricePerLivingSquareMeter')

    # Save the resulting dataframe to a CSV file
    cleaned_data.to_csv('./src/model_data.csv', index=False)


def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


def string_to_int(df, list: list):
    """converts colums in the list from string to int64"""
    columns_to_int64 = list
    df[columns_to_int64] = df[columns_to_int64].astype(float).round().astype('Int64')
