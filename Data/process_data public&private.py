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
import seaborn as sns
#warnings.filterwarnings("ignore")

df=pd.read_csv('account1.csv')
#df['month_time'] = pd.to_datetime(df['month_time'], format='%b-%y')
plt.figure(figsize=(10, 6))

# Plot each feature
plt.plot(df['month_time'], df['n_visit'], marker='o', label='Number of Visits')
plt.plot(df['month_time'], df['n_brand'], marker='o', label='Number of Brands')
plt.plot(df['month_time'], df['n_frt_cate'], marker='o', label='Number of First Categories')
plt.plot(df['month_time'], df['n_mb'], marker='o', label='Number of Mobile Device')
plt.plot(df['month_time'], df['ord_item_qtty'], marker='o', label='Order Item Quantity')
plt.plot(df['month_time'], df['num_loan'], marker='o', linestyle='--', label='Number of Loans')
plt.plot(df['month_time'], df['num_repay'], marker='o', linestyle='--', label='Number of Repayments')

# Adding labels and title
plt.xlabel('Month')
#plt.ylabel('Count')
plt.title('Examples of consumer records')
plt.legend()
#plt.grid(True)

# Format x-axis
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.show()

################################################private  DATA #######################################################

####gender, age, geography
data=pd.read_csv('new_status_by_pin.csv')
province_mapping = {
    "广东省": "Guangdong",
    "四川省": "Sichuan",
    "河南省": "Henan",
    "山东省": "Shandong",
    "江苏省": "Jiangsu",
    "湖北省": "Hubei",
    "河北省": "Hebei",
    '安徽省': 'Anhui',
    '湖南省': 'Hunan',
    '浙江省':'Zhejiang',
    '陕西省':'Shanxi',
    '广西壮族自治区':'Guangxi',
    '江西省':'Jiangxi',
    '福建省':'Fujian',
    '辽宁省':'Liaoning',
    '山西省':'Shanxi',
    '北京市':'Beijing',
    '上海市':'Shanghai',
    '黑龙江省':'Heilongjiang',
    '重庆市':'Chongqing',
    '甘肃省':'Gansu',
    '内蒙古自治区':'Neimenggu',
    '吉林省':'Jilin',
    '天津市':'Tianjin',
    '贵州省':'Guizhou',
    '云南省':'Yunnan',
    '新疆维吾尔自治区':'Xinjiang',
    '海南省':'Hainan',
    '宁夏回族自治区':'Ningxia',
    '青海省':'Qinghai',
    '西藏自治区':'Xizang'
}

# 使用映射字典替换中文省份名称为英文名称
data['pin_id_card_prov'] = data['pin_id_card_prov'].replace(province_mapping)


location_counts=data['pin_id_card_prov'].value_counts()
plt.figure(figsize=(9, 8))
#sns.set_theme(style="white",font='Times New Roman')
#plt.rcParams['font.sans-serif']=['SimHei']
location_counts.plot(kind='bar',color='#4682B4')
plt.title('Geographical Distribution of Consumers')
plt.xlabel('Location')
plt.ylabel('Number of Consumers')
plt.xticks(rotation=45, ha='right')
#plt.tight_layout()
plt.show()

gender=data['pin_gender']
gender = gender.map({0: 'Female', 1: 'Male'})
gender_counts = gender.value_counts()

plt.figure(figsize=(10, 6))
gender_counts.plot(kind='bar', color=['#E7B9B9', '#9EB4D2'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')
plt.xticks(rotation=0)
plt.show()

# 绘制饼图
colors=['#9EB4D2','#E7B9B9']
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, colors=colors,autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution')
plt.show()

age=data['pin_birth_yr']
current_year = datetime.datetime.now().year
# 计算年龄
data['age'] = current_year - data['pin_birth_yr']
import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'

# 绘制核密度估计图
plt.figure(figsize=(10, 6))
sns.kdeplot(data['age'], shade=True)
plt.xlabel('Age',fontsize=15)
plt.ylabel('Density',fontsize=15)
plt.title('Age Density of Concsumers')
plt.grid(False)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(data['age'], bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.title('Age Distribution of Users')
plt.grid(True)
plt.show()


###############################################Public DATA ###################################################
save_path = r'D:\cq\transfer\Revise Experiment\Data\public'
file_path = r'D:\cq\transfer\Revise Experiment\Data\public'
if not os.path.exists(save_path):
    os.makedirs(save_path)
data=pd.read_csv(os.path.join(file_path,f'public-credit.csv'))
data.drop(columns=['ID'])


#生成idx
depth = 8
ID = []
for i in range(int(len(data) / depth)):
        print(i)
        ID.append([i] * depth)
ID = np.array(ID).ravel()
ID = pd.DataFrame(ID)
ID.columns = ['idx']
ID.index = range(len(ID))
allX = pd.concat([ID,data], axis=1)

###生成y
subdata = data[['Customer_ID', 'Month', 'Credit_Score']]
allY = pd.concat([ID, subdata], axis=1)

group_df = list(allY.groupby(['idx']))
firsty = pd.DataFrame()  # predict using first data
secondy = pd.DataFrame()  # predict y using first two data
for i in range(len(group_df)):
    print(i)
    group_df[i][1].index = range(len(group_df[i][1]))
    first = group_df[i][1].drop(index=[0])
    second = group_df[i][1].drop(index=[0, 1])
    firsty = pd.concat([firsty, first])
    secondy = pd.concat([secondy, second])

firsty.index = range(len(firsty))
secondy.index = range(len(secondy))

secondy.to_csv(os.path.join(save_path, 'public_second_y.csv'), index=False, encoding='utf-8-sig')
firsty.to_csv(os.path.join(save_path, 'public_first_y.csv'), index=False, encoding='utf-8-sig')
allX.to_csv(os.path.join(save_path, 'public_x.csv'), index=False, encoding='utf-8-sig')
