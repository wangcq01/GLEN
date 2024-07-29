from dateutil.relativedelta import relativedelta
from copy import deepcopy
import numpy as np
import pandas as pd
import os
from datetime import datetime
import calendar
import datetime
from multiprocessing import Pool
import warnings
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


###############################################Public DATA ###################################################
import pandas as pd

# 定义函数来将 'years and months' 格式转换为总月数
def years_and_months_to_months(years_and_months):
    if pd.isna(years_and_months):
        return pd.NA
    parts = years_and_months.split()
    years = int(parts[0])
    months = int(parts[3])
    total_months = years * 12 + months
    return total_months

def fill_missing_values(series):
    for i in range(len(series)):
        if pd.isna(series.iloc[i]):
            #if i==0 and not pd.isna(series.iloc[i + 1]):
             #   series.iloc[0] = series.iloc[i + 1] - 1
            #elif i==0 and not pd.isna(series.iloc[i + 2]):
               # series.iloc[0] = series.iloc[i + 2] - 2
            #elif i==0 and not pd.isna(series.iloc[i + 3]):
              #  series.iloc[0] = series.iloc[i + 3] - 3
            if i > 0 and not pd.isna(series.iloc[i - 1]):
                series.iloc[i] = series.iloc[i - 1] + 1
            elif i < len(series) - 1 and not pd.isna(series.iloc[i + 1]):
                series.iloc[i] = series.iloc[i + 1] - 1
            elif i < len(series) - 1 and not pd.isna(series.iloc[i + 2]):
                series.iloc[i] = series.iloc[i + 2] - 2
            elif i < len(series) - 1 and not pd.isna(series.iloc[i + 3]):
                series.iloc[i] = series.iloc[i + 3] - 3
            elif i < len(series) - 1 and not pd.isna(series.iloc[i + 4]):
                series.iloc[i] = series.iloc[i + 4] - 4
            #elif i > 0 and not pd.isna(series.iloc[i - 2]):
             #   series.iloc[i] = series.iloc[i - 1] + 2
    return series




def remove_outliers(x):
    try:
        # Convert to numeric
        x = pd.to_numeric(x, errors='raise')

        # Handle negative values
        if x < 0:
            return np.nan
        else:
            if x >= 100:
                return np.nan
            else:
                return x
    except ValueError:
        return np.nan
save_path = r'D:\cq\transfer\Revise Experiment\Data\public'
file_path = r'D:\cq\transfer\Revise Experiment\Data\public'
if not os.path.exists(save_path):
    os.makedirs(save_path)
data=pd.read_csv(os.path.join(file_path,f'public_x.csv'))
data.drop(columns=['ID'])

##processing missing value, abnormal value
data=data.drop(['ID','Customer_ID',	'Month','Name',	'Age','SSN','Occupation','Type_of_Loan'],axis=1)
data['Annual_Income']=data['Annual_Income'].str.rstrip('_').astype(float)
data['Num_of_Loan']=data['Num_of_Loan'].str.rstrip('_').astype(float)
data['Num_of_Delayed_Payment']=data['Num_of_Delayed_Payment'].str.rstrip('_').astype(float)
data['Outstanding_Debt']=data['Outstanding_Debt'].str.rstrip('_').astype(float)
mapping={'Good':1,'Bad':0,'Standard':2}
data['Credit_Mix']=data['Credit_Mix'].replace(mapping)
data['Credit_History_Age'] = data['Credit_History_Age'].apply(years_and_months_to_months)

mapping={'Yes':1,'No':0,'NM':0}
data['Payment_of_Min_Amount']=data['Payment_of_Min_Amount'].replace(mapping)

mapping={'High_spent_Large_value_payments':1,'High_spent_Medium_value_payments':2,'High_spent_Small_value_payments':3,
         'Low_spent_Large_value_payments':4,'Low_spent_Medium_value_payments':5,'Low_spent_Small_value_payments':6}
data['Payment_Behaviour']=data['Payment_Behaviour'].replace(mapping)

mapping={'Good':1,'Poor':0,'Standard':2}
data['Credit_Score']=data['Credit_Score'].replace(mapping)

##negative value
data['Delay_from_due_date'] = data.apply(lambda x: remove_outliers(x['Delay_from_due_date']), axis=1)
data['Num_of_Delayed_Payment'] = data.apply(lambda x: remove_outliers(x['Num_of_Delayed_Payment']), axis=1)
data['Num_Bank_Accounts'] = data.apply(lambda x: remove_outliers(x['Num_Bank_Accounts']), axis=1)
data['Num_Credit_Card'] = data.apply(lambda x: remove_outliers(x['Num_Credit_Card']), axis=1)
data['Num_of_Loan'] = data.apply(lambda x: remove_outliers(x['Num_of_Loan']), axis=1)
data['Num_Credit_Inquiries'] = data.apply(lambda x: remove_outliers(x['Num_Credit_Inquiries']), axis=1)
data['Interest_Rate'] = data.apply(lambda x: remove_outliers(x['Num_Credit_Inquiries']), axis=1)
#data['']
#df["Num_of_Delayed_Payment"] = pd.to_numeric(df["Num_of_Delayed_Payment"])

#print(data.info())
data_group=data.groupby(['idx'])
newdata=pd.DataFrame()
i=0
for group in data_group:
    i=i+1
    print(i)
    if i==1513:
        print()
    sub_data = group[1]
    sub_data['Delay_from_due_date']=sub_data['Delay_from_due_date'].fillna(sub_data['Delay_from_due_date'].median())
    sub_data['Num_of_Delayed_Payment'] = sub_data['Num_of_Delayed_Payment'].fillna(sub_data['Num_of_Delayed_Payment'].median())
    sub_data['Num_Bank_Accounts'] = sub_data['Num_Bank_Accounts'].fillna(sub_data['Num_Bank_Accounts'].median())
    sub_data['Num_Credit_Card'] = sub_data['Num_Credit_Card'].fillna(sub_data['Num_Credit_Card'].median())
    sub_data['Num_of_Loan'] = sub_data['Num_of_Loan'].fillna(sub_data['Num_of_Loan'].median())
    sub_data['Num_Credit_Inquiries'] = sub_data['Num_Credit_Inquiries'].fillna(sub_data['Num_Credit_Inquiries'].median())
    sub_data['Interest_Rate'] = sub_data['Interest_Rate'].fillna(sub_data['Interest_Rate'].median())
    sub_data['Monthly_Inhand_Salary'] = sub_data['Monthly_Inhand_Salary'].fillna(sub_data['Monthly_Inhand_Salary'].median())
    #sub_data['Monthly_Inhand_Salary'] = sub_data['Monthly_Inhand_Salary'].fillna(sub_data['Monthly_Inhand_Salary'].median())
    sub_data['Changed_Credit_Limit'].replace('_', pd.NA, inplace=True)
    sub_data['Changed_Credit_Limit'] = sub_data['Changed_Credit_Limit'].fillna(sub_data['Changed_Credit_Limit'].median())
    sub_data['Credit_Mix'].replace('_', pd.NA, inplace=True)
    sub_data['Credit_Mix'] = sub_data['Credit_Mix'].fillna(sub_data['Credit_Mix'].median())
    sub_data['Credit_History_Age'] = fill_missing_values(sub_data['Credit_History_Age'])
    if (sub_data['Credit_History_Age'].isnull().sum()>0):
        print(sub_data['Credit_History_Age'])
    sub_data['Amount_invested_monthly'].replace('__10000__', pd.NA,inplace=True)
    sub_data['Amount_invested_monthly'] = sub_data['Amount_invested_monthly'].fillna(sub_data['Amount_invested_monthly'].median())
    sub_data['Payment_Behaviour'].replace('!@9#%8', pd.NA,inplace=True)
    sub_data['Payment_Behaviour'] = sub_data['Payment_Behaviour'].fillna(sub_data['Payment_Behaviour'].median())
    sub_data['Monthly_Balance'].replace('__-333333333333333333333333333__', pd.NA,inplace=True)
    sub_data['Monthly_Balance'] = sub_data['Monthly_Balance'].fillna(sub_data['Monthly_Balance'].median())
    newdata = pd.concat([newdata, sub_data])

newdata.index = range(len(newdata))
print(newdata.info())
print(newdata.isnull().sum())
newdata['Changed_Credit_Limit'] =newdata['Changed_Credit_Limit'].astype('float64')
newdata['Credit_Mix']=newdata['Credit_Mix'].astype('int')
newdata['Amount_invested_monthly']=newdata['Amount_invested_monthly'].astype('float64')
newdata['Payment_Behaviour']=newdata['Payment_Behaviour'].astype('int64')
newdata['Monthly_Balance']=newdata['Monthly_Balance'].astype('float64')


print()

cols=['idx','Delay_from_due_date','Num_of_Delayed_Payment','Credit_History_Age','Payment_Behaviour','Num_Bank_Accounts',
      'Num_Credit_Card','Num_of_Loan','Num_Credit_Inquiries','Payment_of_Min_Amount','Credit_Score','Monthly_Inhand_Salary','Credit_Utilization_Ratio',
      'Changed_Credit_Limit','Amount_invested_monthly','Monthly_Balance','Outstanding_Debt','Interest_Rate','Total_EMI_per_month',
      'Annual_Income','Credit_Mix']
finalnewdata=newdata[cols]
finalnewdata.to_csv(os.path.join(save_path, 'new_public_train.csv'), index=False, encoding='utf-8-sig')
