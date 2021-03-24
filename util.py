import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def engineer_features(data):
    """
    Applies changes to certain columns; adds some and deletes others
    :param data: dataframe
    :return: modified dataframe
    """
    ## CREATE BOOLEAN VERSIONS OF BINARY VARIABLES
    data['MALE'] = data['SEX'].apply(lambda f: 1 if(f == 'M') else 0)
    data['SINGLE_PAR'] = data['PARENT1'].apply(lambda f: 1 if (f == 'Yes') else 0)
    data['MARRIED'] = data['MSTATUS'].apply(lambda f: 1 if (f == 'Yes') else 0)
    data['REDCAR'] = data['RED_CAR'].apply(lambda f: 1 if (f == 'yes') else 0)
    data['L_REVOKED'] = data['REVOKED'].apply(lambda f: 1 if (f == 'Yes') else 0)
    data['PRIVATE_USE'] = data['CAR_USE'].apply(lambda f: 1 if (f == 'Private') else 0)
    data['URBAN'] = data['URBANICITY'].apply(lambda f: 1 if (f == 'Highly Urban/ Urban') else 0)

    ## CLEAN UP DOLLAR COLUMNS AND CONVERTING TO NUMBERS
    for col in ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM']:
        data[col] = data[col].str.replace(',', '')
        data[col] = data[col].str.replace('$', '')
        data[col] = pd.to_numeric(data[col])

    ## CREATE BINARY VERSIONS OF SOME DISCRETE VARIABLES
    data['DRIVINGKIDS'] = data['KIDSDRIV'].apply(lambda f: 1 if (f >= 1) else 0)
    # because there is very little difference between 1, 2 or 3 kids driving the car
    data['KIDS'] = data['HOMEKIDS'].apply(lambda f: 1 if (f >= 1) else 0)
    # because people drive as carefully whether they have 1 or 2 kids
    data['PAST_CLAIM'] = data['CLM_FREQ'].apply(lambda f: 1 if (f >= 1) else 0)
    data['NEW_JOB'] = data['YOJ'].apply(lambda f: 1 if (f <= 3) else 0)
    data['YOUNG'] = data['AGE'].apply(lambda f: 1 if (f <= 25) else 0)
    data['HOMEOWNER'] = data['HOME_VAL'].apply(lambda f: 1 if (float(f) > 0) else 0)

    ## FILL IN EMPTY SLOTS
    median_fill = ['INCOME', 'HOME_VAL', 'CAR_AGE']
    for col in median_fill:
        data[col] = data[col].fillna(data[col].median())

    most_freq_fill = ['NEW_JOB', 'YOUNG']
    for col in most_freq_fill:
        data[col] = data[col].fillna(data[col].mode().iloc[0])

    # data['JOB'] = data[['JOB']].apply(lambda f: f.fillna(random.choice(f.dropna())))
    data['JOB'] = data['JOB'].fillna('Unknown')

    ## ENCODING CATEGORICAL VARIABLES (changes motivated by convenience or by crashrates; see crashrates functions)
    # College education or not
    data['COLLEGE_ED'] = data['EDUCATION'].apply(lambda f: 0 if ((f == 'z_High School') or (f == '<High School')) else 1)
    # Minivan drivers have a significantly lower crashrate
    data['MINIVAN'] = data['CAR_TYPE'].apply(lambda f: 1 if (f == 'Minivan') else 0)
    # Jobs can be seperated into three categories
    data['HIGH_RISK_JOB'] = data['JOB'].apply(lambda f: 1 if ((f == 'Student') or (f == 'z_Blue Collar')) else 0)
    data['MID_RISK_JOB'] = data['JOB'].apply(lambda f: 1 if ((f == 'Clerical') or (f == 'Home Maker')
                                                             or (f == 'Unknown') or (f == 'Professional')) else 0)
    data['LOW_RISK_JOB'] = data['JOB'].apply(lambda f: 1 if ((f == 'Doctor') or (f == 'Lawyer') or (f == 'Manager')) else 0)

    ## DELETE OLD VERSIONS
    del data['SEX']
    del data['PARENT1']
    del data['MSTATUS']
    del data['RED_CAR']
    del data['REVOKED']
    del data['CAR_USE']
    del data['URBANICITY']
    del data['KIDSDRIV']
    del data['HOMEKIDS']
    del data['CLM_FREQ']
    del data['YOJ']
    del data['EDUCATION']
    del data['AGE']
    del data['JOB']
    del data['CAR_TYPE']

    del data['INDEX']

    return data


def crashrates_age(data):
    """
    Computes and displays crashrates for 3 age categories
    :param data: dataframe
    :return: //
    """
    yng = 0
    yng_crash = 0
    mid = 0
    mid_crash = 0
    old = 0
    old_crash = 0

    for i, val in enumerate(data['AGE']):
        if val <= 25:
            yng += 1
            if data['TARGET_FLAG'][i] == 1:
                yng_crash += 1
        elif 25 < val < 60:
            mid += 1
            if data['TARGET_FLAG'][i] == 1:
                mid_crash += 1
        else:
            old += 1
            if data['TARGET_FLAG'][i] == 1:
                old_crash += 1

    print("Young (-25):", yng_crash / yng)
    print("Middle-aged (26-59):", mid_crash / mid)
    print("Old (60+):", old_crash / old)


def crashrates_categorical_var(data, var_name):
    """
    Computes and displays crashrates for each mode of a given categorical variable
    :param data: dataframe
    :param var_name: name of the categorical variable to be studied
    :return: //
    """
    unique = data[var_name].unique()
    dic = {cat:[0, 0] for cat in unique}

    for i, val in enumerate(data[var_name]):
        dic[val][1] += 1  # increment number of customers with that type of car
        if data['TARGET_FLAG'][i]:
            dic[val][0] += 1  # increment number of crashes among those

    for cat in unique:
        print('%s: %f' % (cat, dic[cat][0] / dic[cat][1]))


def get_n_rows(df):
    """
    Get number of rows
    :param df: dataframe
    :return: Number of rows
    """
    n_rows = df['INDEX'].count()

    return n_rows


def get_incomplete_cols(df):
    """
    Indicates which columns have missing data
    :param df: dataframe
    :return: list of columns where data is missing
    """
    missing_list = []
    n_rows = get_n_rows(df)
    for i, col in enumerate(df.columns):
        non_na = df[col].count()
        if non_na != n_rows:
            missing_list.append(col)

    return missing_list


def split_data(data):
    """
    Splits the dataset into train and test sets
    :param data: dataset
    :return: training examples; test examples; training targets; test targets
    """

    y_trainflag = data.pop('TARGET_FLAG')
    del data['TARGET_AMT']  # we don't use TARGET_AMT in this study
    x_train, x_test, y_train, y_test = train_test_split(data, y_trainflag, test_size=0.25, shuffle=False)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    return x_train, x_test, y_train, y_test


def format_test_data(data):
    """
    Formats test data appropriately
    :param data: test dataframe
    :return: modified test data
    """
    del data['TARGET_FLAG']
    del data['TARGET_AMT']

    return data


def print_conf_mat(cm):
    """
    Prints confusion matrices nicely
    :param cm: confusion matrix
    :return: //
    """
    labels = ['0', '1']
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    print('\nConfusion Matrix:\n')

    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print(' ')
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            print(cell, end=" ")
        print()


def get_limited_corr(full_data, selected_headers):
    """
    Plot correlation matrix for a subset of columns
    :param full_data: full dataframe
    :param selected_headers: list of the names of the columns to use
    :return: //
    """
    selected_data = [full_data[el] for el in selected_headers]
    temp_df = pd.concat(selected_data, axis=1, keys=selected_headers)
    print(temp_df.info())
    plot_corr(temp_df)


def get_corr(data):
    """
    Computes correlation matrix from the columns of the dataset
    :param data: dataset
    :return: correlation matrix
    """
    return data.corr()


def plot_corr(data):
    """
    Plots the correlation matrix
    :param data: dataset
    :return: //
    """
    corr = get_corr(data)

    fig, ax = plt.subplots()
    ax.matshow(corr, cmap='coolwarm')
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    plt.xticks(np.arange(len(data.columns)), data.columns.values, rotation=35)
    plt.yticks(np.arange(len(data.columns)), data.columns.values)
    plt.show()


def get_best_classifier(clsf, params):
    """
    Calls GridSearchCV
    :param clsf: classifier to study
    :param params: dictionary containing the parameters to test
    :return: best classifier
    """
    best_classifier = GridSearchCV(clsf, params, scoring='neg_mean_squared_error')

    return best_classifier


def explore_params(train, y_train):
    """
    Bigger function to use GridSearchCV. Calls get_best_classifier()
    :param train: training examples
    :param y_train: training targets
    :return: //
    """
    n_estimators = {"n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]}
    # n_estimators = {"n_estimators": [400, 450, 500, 550, 600, 650, 700]}
    param = n_estimators

    clsf = XGBClassifier(use_label_encoder=False, eval_metric='error')
    best_clsf = get_best_classifier(clsf, params=param)
    best_clsf.fit(train, y_train)

    plt.subplot(121)
    plt.plot(list(param.values())[0], best_clsf.cv_results_['mean_test_score'], 'o-')
    plt.xlabel(list(param.keys())[0])
    plt.ylabel("mean")
    plt.subplot(122)
    plt.plot(list(param.values())[0], best_clsf.cv_results_['std_test_score'], 'o-')
    plt.xlabel(list(param.keys())[0])
    plt.ylabel("std")
    plt.tight_layout()
    plt.show()