import math
import pandas as pd
import numpy as np
import datetime
import sklearn.tree
#from graphviz import Source
#from IPython.display import SVG
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import RFE
import os

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier#, ExtraTreesClassifier
#from sklearn.tree import DecisionTreeClassifier
#import xgboost as xgb

#import matplotlib.pyplot as plt
#%matplotlib inline
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

def transform_rating(rating):
    '''Input: Column of Data with NPS Field (on a scale of 1 to 5...)
    Output: Column with either Promoter/Detractor/Passive label, or the corresponding weights, based on datatype'''
    
    if rating == 5:
        return 100
    elif rating <= 3:
        return -100
    elif rating == 4:
        return 0
    else:
        return np.nan

def make_AB_dataset(df,prefixes):
    '''df: initial input dataframe of flight data
    prefixes: flights have prefixes - this arg is a list for which prefixes we want to filter on'''
    df = df[df['AudienceGroup']=='Production']
    df['Date'] = pd.to_datetime(df['Date'])
    flights = df.FlightId.astype(str).unique()
    controls = [f for f in flights if f.endswith('-c') or f.__contains__('control')]
    treatments = [f for f in flights if f.endswith('-t') or f.__contains__('treatment')]
    neither = [f for f in flights if (f not in controls) and (f not in treatments)]
    union = {'controls':[c.rstrip('control') for c in controls],'treatments':[t.rstrip('treatments') for t in treatments]}
    no_control = list(set(union['treatments'])-set(union['controls']))
    control_treatment_pairs = list(set(union['treatments'])-set(no_control))
    df['FlightPair']=df['FlightId'].astype(str).map(lambda x: x.rstrip('control'))
    df['FlightPair']=df['FlightPair'].map(lambda x: x.rstrip('treatment'))
    #df['FlightPair'].replace('docowner-canary','canary-docowner',inplace=True)
    
    
    ab_df = df[df.FlightId.notnull()]
    ab_df = ab_df.sort_values(by='Date')
    ab_df.drop_duplicates(keep='last', inplace=True)
    print(ab_df.shape, ' before filtering out non-pairs')
    ab_df = ab_df[ab_df['FlightPair'].isin(control_treatment_pairs)]
    print(ab_df.shape, ' after filtering out non-pairs')
    ab_df.loc[ab_df.FlightId.str.endswith('-c'),'Group'] = 'Control'
    ab_df.loc[ab_df.FlightId.str.endswith('control'),'Group'] = 'Control'
    ab_df.loc[ab_df.FlightId.str.endswith('-t'),'Group'] = 'Treatment'
    ab_df.loc[ab_df.FlightId.str.endswith('treatment'),'Group'] = 'Treatment'
    ab_df.loc[ab_df.FlightId.str.endswith('-c'),'Flight'] = 0
    ab_df.loc[ab_df.FlightId.str.endswith('control'),'Flight'] = 0
    ab_df.loc[ab_df.FlightId.str.endswith('-t'),'Flight'] = 1
    ab_df.loc[ab_df.FlightId.str.endswith('treatment'),'Flight'] = 1
    ab_df = ab_df[ab_df.Flight.notnull()]
    ab_df['NPS'] = ab_df['Rating'].apply(transform_rating)
    value_key = ab_df.sort_values(by='Date').groupby(['OcvId'])['NPS'].last().to_dict()
    exp_df = ab_df.groupby(['OcvId','FlightPair'])['Flight'].last().unstack()
    print('Feature Matrix should have ',ab_df.OcvId.nunique(), ' rows and ',ab_df.FlightPair.nunique(),' columns')
    print('Final Shape:',exp_df.shape)
    if prefixes: #i.e. if the input list is empty:
        for p in prefixes:
            exp_df = exp_df.iloc[:,exp_df.columns.str.startswith(p)]
    exp_df['NPS'] = exp_df.index.map(value_key)
    return exp_df.fillna(0)

def get_flight_durations():
    ''' no inputs/arguments, just make sure you have all of the Tabular Flight data files you are using.'''
    df = pd.concat([Excel_df[Excel_df['AudienceGroup']=='Production'][Excel_df.FlightId.notnull()],
                    Word_df[Word_df['AudienceGroup']=='Production'][Word_df.FlightId.notnull()],
                    PP_df[PP_df['AudienceGroup']=='Production'][PP_df.FlightId.notnull()]])
    #df = df[df['AudienceGroup']=='Production']
    df['Date'] = pd.to_datetime(df['Date'])
    flights = df.FlightId.astype(str).unique()
    controls = [f for f in flights if f.endswith('-c') or f.__contains__('control')]
    treatments = [f for f in flights if f.endswith('-t') or f.__contains__('treatment')]
    neither = [f for f in flights if (f not in controls) and (f not in treatments)]
    union = {'controls':[c.rstrip('control') for c in controls],'treatments':[t.rstrip('treatments') for t in treatments]}
    no_control = list(set(union['treatments'])-set(union['controls']))
    control_treatment_pairs = list(set(union['treatments'])-set(no_control))
    df['FlightPair']=df['FlightId'].astype(str).map(lambda x: x.rstrip('control'))
    df['FlightPair']=df['FlightPair'].map(lambda x: x.rstrip('treatment'))
    
    print('Flight Pairs Assigned')
    
    ab_df = df[df.FlightPair.notnull()]
    #ab_df.drop_duplicates(keep='last', inplace=True)
    ab_df = ab_df[ab_df['FlightPair'].isin(control_treatment_pairs)]
    flight_starts = ab_df.sort_values(by='Date').groupby(['FlightPair'])['Date'].first()
    flight_ends = ab_df.sort_values(by='Date').groupby(['FlightPair'])['Date'].last()
    flight_durations = pd.concat([flight_starts,flight_ends],axis=1)
    flight_durations.columns = ['FlightStart','FlightEnd']
    return flight_durations

month='AsOf'+'November'
data_needs_concatenation = True

print('Making Datasets')
if data_needs_concatenation == True:
    Word_df1 = pd.read_csv('Word_updated_dataOct.tsv', sep='\t')
    Word_df2 = pd.read_csv('Word_updated_dataNov.tsv',sep='\t')
    Word_df = pd.concat([Word_df1,Word_df2], axis=0)

    Excel_df1 = pd.read_csv('Excel_updated_dataOct.tsv', sep='\t')
    Excel_df2 = pd.read_csv('Excel_updated_dataNov.tsv',sep='\t')
    Excel_df = pd.concat([Excel_df1,Excel_df2])

    PP_df1 = pd.read_csv('PowerPoint_updated_dataOct.tsv', sep='\t')
    PP_df2 = pd.read_csv('PowerPoint_updated_dataNov.tsv',sep='\t')
    PP_df = pd.concat([PP_df1,PP_df2])
else:
    Word_df =  Word_df1 = pd.read_csv('Word_updated_data.tsv', sep='\t')
    Excel_df1 = pd.read_csv('Excel_updated_data.tsv', sep='\t')
    PP_df1 = pd.read_csv('PowerPoint_updated_data.tsv', sep='\t')




print(pd.to_datetime(Word_df['Date']).min(),pd.to_datetime(Word_df['Date']).max())
print('Getting Flight Durations (Within Span of dates above)')
flight_durations = get_flight_durations()
flight_durations.to_csv('Flights'+month+'.csv')

excel_df = make_AB_dataset(Excel_df,[])
word_df = make_AB_dataset(Word_df,[])
pp_df = make_AB_dataset(PP_df,[])

excel_flights = list(excel_df.columns)
word_flights = list(word_df.columns)
common_flights = list(set(excel_flights).intersection(word_flights))
pp_flights = list(pp_df.columns)
common_flights = list(set(common_flights).intersection(pp_flights))

common_flights.remove('NPS')
dc = ['docowner-canary-','canary-docowner-','canary2','canary-']


X = {'excel':excel_df.iloc[:,excel_df.columns.str.startswith('xls')],
     'word':word_df.iloc[:,word_df.columns.str.startswith('wac')],
     'pp':pp_df.iloc[:,pp_df.columns.str.startswith('pp')],
     'all':pd.concat([excel_df,word_df,pp_df]).loc[:,common_flights]}
y = {'excel':excel_df['NPS'].replace([100,0],1).replace(-100,0)
     ,'word':word_df['NPS'].replace([100,0],1).replace(-100,0),
     'pp':pp_df['NPS'].replace([100,0],1).replace(-100,0),
     'all':pd.concat([excel_df,word_df,pp_df])['NPS'].replace([100,0],1).replace(-100,0)}


print('Excel: ',X['excel'].shape, y['excel'].shape)
print('Word: ',X['word'].shape, y['word'].shape)
print('PowerPoint: ',X['pp'].shape, y['pp'].shape)
print('All: ',X['all'].shape, y['all'].shape)

model_perfs = pd.DataFrame(columns=['Logistic','RandomForest','GradientBoosting'])
pd.DataFrame(data=['StatSigFlights']).to_csv('StatSigFlights'+month+'.csv',index=False,header=None)

# print('Optional Step: Train LR vs RF vs GB')
# for key in ['excel','word','pp','all']:
# ########## WHEN PREDICTORS ARE BINARY (1 for Treatment, 0 otherwise) ###############
#     logit = LogisticRegression().fit(X[key],y[key])
#     rf = RandomForestClassifier(random_state=0).fit(X[key],y[key])
#     gb = xgb.XGBClassifier(random_state=0, n_jobs=4).fit(X[key],y[key])
    
#     log_acc = round(logit.score(X[key],y[key]),3)
#     rf_acc = round(rf.score(X[key],y[key]),3)
#     gb_acc = round(gb.score(X[key],y[key]),3)

#     model_perfs.loc[key+'Accuracy']=[log_acc,rf_acc,gb_acc]


# model_perfs.to_csv('ABModelTrainingPerformances.csv')


for key in ['excel','word','pp','all']:
    X_train, X_test, y_train, y_test = train_test_split(X[key], y[key], test_size=0.2)

    rf = RandomForestClassifier(random_state=0)
    param_grid = { 
        'n_estimators': [50, 100, 250, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [2,5,10,None],
        'oob_score': [True,False]}

    CV_rfc = GridSearchCV(estimator=rf, n_jobs=-1, param_grid=param_grid, verbose=10, scoring='neg_log_loss',cv= 5)
    CV_rfc.fit(X_train, y_train)
    print(CV_rfc.best_params_, CV_rfc.best_score_)

    print(RandomForestClassifier(random_state=0).fit(X_train,
                                            y_train).score(X_test,y_test))


    rfc = RandomForestClassifier(**CV_rfc.best_params_)
    rfc.fit(X_train,y_train)
    print(rfc.score(X_test,y_test))

    learners = rfc.feature_importances_.argsort()[::-1]

    features = pd.DataFrame(X_train.columns[learners], rfc.feature_importances_[learners])
    features = features[features.index>0.025]
    features

    print(LogisticRegression().fit(X_train,y_train).score(X_test,y_test), ' -->...')


    rfe = RFE(logit,10)
    rfe = rfe.fit(X_train,y_train.values.ravel())

    rfe.ranking_

    #identified columns Recursive Feature Elimination
    idc_rfe = pd.DataFrame({"rfe_support" :rfe.support_,
                        "columns" : [i for i in X_train.columns],
                        "ranking" : rfe.ranking_,
                        })

    cols = idc_rfe[idc_rfe["rfe_support"] == True]["columns"].tolist()

    cols.extend(features.FlightPair.values.tolist())
    cols = list(set(cols))

    logit = sm.Logit(y[type],X[type].loc[:,cols])
    flogit = logit.fit()
    print(flogit.summary())

    coefficients = flogit.summary2().tables[1]
    coefficients = coefficients[coefficients['P>|z|']<0.1]
    coefficients['Odds Ratio']=np.exp(coefficients['Coef.'])
    coefficients['O.R.LB']=np.exp(coefficients['[0.025'])
    coefficients['O.R.UB']=np.exp(coefficients['0.975]'])
    coefficient['Probability'] = coefficients['Odds Ratio'].round(1)*0.5 #- 0.5
    coefficient['Probability'] = coefficients['Probability'].mask(coefficients['Probability']>=1,0.99)
    coefficients.join(flight_durations)
    coefficients.to_csv('StatSigFlights'+month+'.csv',mode='a')
    