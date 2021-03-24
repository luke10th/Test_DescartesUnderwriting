import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

import util

print("--------START--------")
cwd = os.getcwd()
print(cwd)


## LOADING DATA

train_data_file = 'train_auto.csv'
test_data_file = 'test_auto.csv'
og_train_data = pd.read_csv(train_data_file)
og_test_data = pd.read_csv(test_data_file)

n_train_rows = util.get_n_rows(og_train_data)
n_test_rows = util.get_n_rows(og_test_data)


## CHECKING FOR MISSING VALUES

train_missing = util.get_incomplete_cols(og_train_data)
print("\nTRAINDATA- Missing values in:", train_missing)


## FORMATTING COLUMNS

train_data = util.engineer_features(og_train_data)
print(train_data.info())
test_indexes = og_test_data['INDEX']
test_data = util.engineer_features(og_test_data)


## SPLITTING OUR TRAINING DATA INTO A TRAINING SET AND AN EVALUATION SET

x_train, x_eval, y_train, y_eval = util.split_data(train_data)


## EXPLORING PARAMETERS

# util.explore_params(x_train, y_train)


## PREDICTING TARGET_FLAG ON EVAL SET

flag_model = XGBClassifier(n_estimators=300, max_depth=5, use_label_encoder=False, eval_metric='error')
flag_model.fit(x_train, y_train)

pred_y_eval = flag_model.predict(x_eval)


## DISPLAYING RESULTS AND MOTIVATING OUR METRIC CHOICE

conf_mat = confusion_matrix(y_eval, pred_y_eval)
tn, fp, fn, tp = conf_mat.ravel()
print('TN:', tn)
print('FP:', fp)
print('FN:', fn)
print('TP:', tp)

util.print_conf_mat(conf_mat)

# The goal for the insurance company is to flag as accurately as possible which customers are 'risky'
# (i.e. who are more likely to get into car accidents) so as to charge them higher rates
# prior to said crashes

# We want to increase Recall, so as to miss as few risky customers as possible
print('\nRecall:', tp / (tp + fn))

# We also want to have a high Specificity criterion: it means that we reduce the number of people we have
# wrongfully flagged flagged as 'risky'
print('Specificity:', tn / (tn + fp))

# We have to choose one of these two metrics: from a short-term perspective, it is better to prioritize
# Recall over Specificity. We might charge higher rates for customers who do not 'deserve' it,
# and those might want to leave later for another, cheaper insurance company; but in the mean time
# we are making more money off of our current clients.


## DISPLAYING FINAL RESULTS

print('FINAL RESULT (Recall criterion): %f %%' % (100*tp / (tp + fn)))
print('(see comments for explanation)\n')
# Better results can be obtained by fine-tuning the hyperparameters of the classifier, using the
# explore_parameters() function in the util file; however this takes time.


## PREDICTING TARGET_FLAG ON TEST SET

x_test = util.format_test_data(test_data)
pred_y_test = flag_model.predict(x_test)


## WRITING PREDICTION TO .CSV FILE

predictions = pd.DataFrame(list(zip(test_indexes, pred_y_test)), columns=['INDEX', 'TARGET_FLAG'])
predictions.to_csv(os.path.join(cwd, "predictions.csv"), index=False)


print("---------END---------")
