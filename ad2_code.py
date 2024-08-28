import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pysanky import sankey

# For the predictive models
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBSklearn
#from xgboost import XGBClassifier as XGB
#import lightgbm as lgb

# Removing annoying warnings
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def count_by_group(data, feature, target):
    df = data.groupby([feature, target])[target].agg(['count'])
    temp = data.groupby([feature])[target].agg(['count'])
    df['pct'] = 100*df.div(temp, level = feature).reset_index()['count'].values
    return df.reset_index()

def classification_report_to_dataframe(true, predictions, predictions_proba, model_name, balanced = 'no'):
    a = classification_report(true, predictions, output_dict = True)
    zeros = pd.DataFrame(data = a['0'], index = [0]).iloc[:,0:3].add_suffix('_0')
    ones = pd.DataFrame(data = a['1'], index = [0]).iloc[:,0:3].add_suffix('_1')
    df = pd.concat([zeros, ones], axis = 1)
    temp = list(df)
    df['Model'] = model_name
    df['Balanced'] = balanced
    df['Accuracy'] = accuracy_score(true, predictions)
    df['Balanced_Accuracy'] = balanced_accuracy_score(true, predictions)
    df['AUC'] = roc_auc_score(true, predictions_proba, average = 'macro')
    df = df[['Model', 'Balanced', 'Accuracy', 'Balanced_Accuracy', 'AUC'] + temp]
    return df

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
   
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


dataset = pd.read_csv('churn.csv')


dataset.head()


dataset.info()


dataset.isna().sum()


round(dataset.describe(),3)


exited = len(dataset[dataset['Exited'] == 1]['Exited'])
not_exited = len(dataset[dataset['Exited'] == 0]['Exited'])
exited_perc = round(exited/len(dataset)*100,1)
not_exited_perc = round(not_exited/len(dataset)*100,1)

print('Number of clients that have exited the program: {} ({}%)'.format(exited, exited_perc))
print('Number of clients that haven\'t exited the program: {} ({}%)'.format(not_exited, not_exited_perc))


country = list(dataset['Geography'].unique())
gender = list(dataset['Gender'].unique())

print(country)
print(gender)


dataset['Exited_str'] = dataset['Exited']
dataset['Exited_str'] = dataset['Exited_str'].map({1: 'Exited', 0: 'Stayed'})


gender_count = dataset['Gender'].value_counts()
gender_pct= gender_count / len(dataset.index)

gender = pd.concat([gender_count, round(gender_pct,2)], axis=1)\
        .set_axis(['count', 'pct'], axis=1, inplace = True)
gender


gender_count = dataset['Gender'].value_counts()
gender_pct = gender_count / len(dataset.index)

gender = pd.concat([gender_count, round(gender_pct, 2)], axis=1)\
        .set_axis(['count', 'pct'], axis=1, inplace=True)


# In[21]:


gender_count = dataset['Gender'].value_counts()
gender_pct = gender_count / len(dataset.index)

gender = pd.concat([gender_count, round(gender_pct, 2)], axis=1)\
        .rename(columns={0: 'count', 1: 'pct'})
gender


geo_count = dataset['Geography'].value_counts()
geo_pct = geo_count / len(dataset.index)

geo = pd.concat([geo_count, round(geo_pct, 2)], axis=1)\
        .set_axis(['count', 'pct'], axis=1)
geo


def count_by_group(data, feature, target):
    df = data.groupby([feature, target])[target].agg(['count'])
    temp = data.groupby([feature])[target].agg(['count'])
    df['pct'] = 100*df.div(temp, level = feature).reset_index()['count'].values
    return df.reset_index()


count_by_group(dataset, feature = 'Gender', target = 'Exited')


import plotly.graph_objects as go

colorDict = {
    'Exited': '#f71b1b',
    'Stayed': 'grey',
    'Female': '#FFD700',
    'Male': '#8E388E'
}

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=["Male", "Female", "Stayed", "Exited"],
        color=[colorDict['Male'], colorDict['Female'], colorDict['Stayed'], colorDict['Exited']]
    ),
    link=dict(
        source=[0, 0, 1, 1],
        target=[2, 3, 2, 3],
        value=[len(dataset[(dataset['Gender'] == 'Male') & (dataset['Exited_str'] == 'Stayed')]),
               len(dataset[(dataset['Gender'] == 'Male') & (dataset['Exited_str'] == 'Exited')]),
               len(dataset[(dataset['Gender'] == 'Female') & (dataset['Exited_str'] == 'Stayed')]),
               len(dataset[(dataset['Gender'] == 'Female') & (dataset['Exited_str'] == 'Exited')])
              ]
    )
)])

fig.update_layout(title_text="Gender Sankey Diagram")
fig.show()


count_by_group(dataset, feature = 'Geography', target = 'Exited')


import plotly.graph_objects as go

colorDict = {
    'Exited': '#f71b1b',
    'Stayed': 'grey',
    'France': '#f3f71b',
    'Spain': '#12e23f',
    'Germany': '#f78c1b'
}

# Count occurrences of combinations of Geography and Exited_str
data_counts = dataset.groupby(['Geography', 'Exited_str']).size().reset_index(name='count')

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=list(data_counts['Geography'].unique()) + list(data_counts['Exited_str'].unique()),
        color=[colorDict.get(x, 'blue') for x in list(data_counts['Geography'].unique())] + 
              [colorDict.get(x, 'blue') for x in list(data_counts['Exited_str'].unique())]
    ),
    link=dict(
        source=[data_counts['Geography'].unique().tolist().index(geo) for geo in data_counts['Geography']],
        target=[len(data_counts['Geography'].unique()) + data_counts['Exited_str'].unique().tolist().index(ex) 
                for ex in data_counts['Exited_str']],
        value=data_counts['count']
    )
)])

fig.update_layout(title_text="Geography Sankey Diagram")
fig.show()


HasCrCard_count = dataset['HasCrCard'].value_counts()
HasCrCard_pct = HasCrCard_count / len(dataset.index)

HasCrCard = pd.concat([HasCrCard_count, HasCrCard_pct], axis=1)\
        .set_axis(['count', 'pct'], axis=1)
HasCrCard


count_by_group(dataset, feature = 'HasCrCard', target = 'Exited')


# Count occurrences of combinations of 'HasCrCard_str' and 'Exited_str'
plot_data = dataset.groupby(['HasCrCard_str', 'Exited_str']).size().unstack(fill_value=0)

# Define colors for each category
colorDict = {
    'Exited': '#f71b1b',
    'Stayed': 'grey',
    'Has Credit Card': '#FFD700',
    'Does not have Credit Card': '#8E388E'
}

# Plot stacked bar plot
plot_data.plot(kind='bar', stacked=True, color=[colorDict[col] for col in plot_data.columns])

# Set labels and title
plt.xlabel('HasCrCard')
plt.ylabel('Count')
plt.title('HasCrCard vs. Exited')

# Show plot
plt.legend(title='Exited')
plt.show()


IsActiveMember_count = dataset['IsActiveMember'].value_counts()
IsActiveMember_pct = IsActiveMember_count / len(dataset.index)

IsActiveMember = pd.concat([IsActiveMember_count, round(IsActiveMember_pct, 2)], axis=1)\
                  .set_axis(['count', 'pct'], axis=1)

print(IsActiveMember)


count_by_group(dataset, feature = 'IsActiveMember', target = 'Exited')


# Count occurrences of combinations of 'IsActiveMember_str' and 'Exited_str'
plot_data = dataset.groupby(['IsActiveMember_str', 'Exited_str']).size().unstack(fill_value=0)

# Define colors for each category
colorDict = {
    'Exited': '#f71b1b',
    'Stayed': 'grey',
    'Is Active Member': '#FFD700',
    'Is Not ActiveMember': '#8E388E'
}

# Plot stacked bar plot
plot_data.plot(kind='bar', stacked=True, color=[colorDict[col] for col in plot_data.columns])

# Set labels and title
plt.xlabel('IsActiveMember')
plt.ylabel('Count')
plt.title('IsActiveMember vs. Exited')

# Show plot
plt.legend(title='Exited')
plt.show()


import pandas as pd

# Calculate count and percentage of occurrences for each value in 'NumOfProducts' column
NumOfProducts_count = dataset['NumOfProducts'].value_counts()
NumOfProducts_pct = NumOfProducts_count / len(dataset.index)

# Concatenate count and percentage into a DataFrame and set axis labels
NumOfProducts = pd.concat([NumOfProducts_count, round(NumOfProducts_pct, 2)], axis=1)
NumOfProducts.columns = ['count', 'pct']

print(NumOfProducts)


count_by_group(dataset, feature = 'NumOfProducts', target = 'Exited')


import matplotlib.pyplot as plt

# Count occurrences of combinations of 'NumOfProducts_str' and 'Exited_str'
plot_data = dataset.groupby(['NumOfProducts_str', 'Exited_str']).size().unstack(fill_value=0)

# Define colors for each category
colorDict = {
    'Exited': '#f71b1b',
    'Stayed': 'grey',
    '1': '#f3f71b',
    '2': '#12e23f',
    '3': '#f78c1b',
    '4': '#8E388E'
}

# Plot bar plot
plot_data.plot(kind='bar', stacked=True, color=[colorDict[col] for col in plot_data.columns])

# Set labels and title
plt.xlabel('NumOfProducts')
plt.ylabel('Count')
plt.title('NumOfProducts vs. Exited')

# Show plot
plt.legend(title='Exited')
plt.show()

figure = plt.figure(figsize=(15,8))
plt.hist([
        dataset[(dataset.Exited==0)]['Age'],
        dataset[(dataset.Exited==1)]['Age']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
plt.xlabel('Age (years)')
plt.ylabel('Number of customers')
plt.legend()


fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (15,15))
fig.subplots_adjust(left=0.2, wspace=0.6)
ax0, ax1, ax2, ax3 = axes.flatten()

ax0.hist([
        dataset[(dataset.Exited==0)]['CreditScore'],
        dataset[(dataset.Exited==1)]['CreditScore']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax0.legend()
ax0.set_title('Credit Score')

ax1.hist([
        dataset[(dataset.Exited==0)]['Tenure'],
        dataset[(dataset.Exited==1)]['Tenure']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax1.legend()
ax1.set_title('Tenure')
ax2.hist([
        dataset[(dataset.Exited==0)]['Balance'],
        dataset[(dataset.Exited==1)]['Balance']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax2.legend()
ax2.set_title('Balance')

ax3.hist([
        dataset[(dataset.Exited==0)]['EstimatedSalary'],
        dataset[(dataset.Exited==1)]['EstimatedSalary']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax3.legend()
ax3.set_title('Estimated Salary')

fig.tight_layout()
plt.show()


list_cat = ['Geography', 'Gender']
dataset = pd.get_dummies(dataset, columns = list_cat, prefix = list_cat)
dataset.head()


import pandas as pd

# Assuming dataset is defined and populated correctly
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Exited_str', 'HasCrCard_str', 'IsActiveMember_str', 'NumOfProducts_str']

# Check if columns exist in the DataFrame before dropping
columns_exist = all(col in dataset.columns for col in columns_to_drop)

if columns_exist:
    dataset = dataset.drop(columns=columns_to_drop)
    print("Columns dropped successfully.")
else:
    print("One or more columns do not exist in the DataFrame.")


dataset.info()


features = list(dataset.drop('Exited', axis = 1))
target = 'Exited'


train, test = train_test_split(dataset, test_size = 0.2, random_state = 1)

print('Number of clients in the dataset: {}'.format(len(dataset)))
print('Number of clients in the train set: {}'.format(len(train)))
print('Number of clients in the test set: {}'.format(len(test)))


exited_train = len(train[train['Exited'] == 1]['Exited'])
exited_train_perc = round(exited_train/len(train)*100,1)

exited_test = len(test[test['Exited'] == 1]['Exited'])
exited_test_perc = round(exited_test/len(test)*100,1)

print('Complete Train set - Number of clients that have exited the program: {} ({}%)'.format(exited_train, exited_train_perc))
print('Test set - Number of clients that haven\'t exited the program: {} ({}%)'.format(exited_test, exited_test_perc))


from sklearn.preprocessing import StandardScaler

# Assuming train and test are DataFrames and features is a list of column names
features = ['column1', 'column2', 'column3']  # Example list of feature column names

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the training set
train_scaled = train.copy()  # Make a copy to avoid modifying the original DataFrame
train_scaled[features] = scaler.fit_transform(train_scaled[features])

# Transform the test set
test_scaled = test.copy()  # Make a copy to avoid modifying the original DataFrame
test_scaled[features] = scaler.transform(test_scaled[features])

# Optionally, you can overwrite the original DataFrames with the scaled versions
# train = train_scaled
# test = test_scaled


parameters = {'C': [0.01, 0.1, 1, 10],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'max_iter': [50, 100, 150]}
LR = LogisticRegression(penalty = 'l2')
model_LR = GridSearchCV(LR, parameters, cv = 5, n_jobs = 10, verbose = 1).fit(train[features], train[target])
pd.DataFrame(model_LR.cv_results_)