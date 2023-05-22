# kaggle_wine

- fixed acidity : 고정 산도
- volatile acidity: 휘발성 산성
- citric acid: 구연산
- residual sugar: 잔류 설탕
- chlorides: 염화물
- free sulfur dioxide: 유리 인산화황
- total sulfur dioxide: 총 이산화황
- density: 밀도
- pH: 
- sulphates	alcohol: 황산염 알코올
- quality

### 파일 확인
![image](https://user-images.githubusercontent.com/120009186/234346345-3d1ef9cf-b382-4154-b597-7e7cc097f57d.png)

### EDA

train.iloc[:,:-1].describe().T.sort_values(by='mean',ascending=False)\
                     .style.background_gradient(cmap='GnBu')\
                     .bar(subset=["max"], color='#BB0000')\
                     .bar(subset=["mean",], color='green')
![image](https://user-images.githubusercontent.com/120009186/234347031-4d8e4579-42f0-47e6-af87-c3aa81d7237a.png)

ncol = 4
nrow = len(cont_features)

fig, axes = plt.subplots(nrow, ncol, figsize=(18, 5*nrow))

for r in range(nrow):
    row = cont_features[r]
    
    sns.histplot(train[row], ax = axes[r,0],color='#F8766D', label='Train data' ,
                 fill =True)
    sns.histplot(test[row], ax=axes[r,0],color='#00BFC4', label='TEST data' ,
                 fill =True)
    sns.kdeplot(train[row],ax=axes[r,1],color='#F8766D', label='Train data' ,
                fill =True)
    sns.kdeplot(test[row],ax=axes[r,1],color='#00BFC4', label='TEST data' ,
                fill =True)
    sns.boxplot(x = train[row],y=train[target],ax=axes[r,2],color='#F8766D',
                orient = "h")
    sns.boxplot(x = test[row],ax=axes[r,3],color='#00BFC4', orient = "h")
    
    axes[r,0].legend()
    axes[r,1].legend()
    axes[r,0].title.set_text("Histogram Plot")
    axes[r,1].title.set_text("Distribution Plot")
    axes[r,2].title.set_text("Box Plot- Train Data")
    axes[r,3].title.set_text("Box Plot- Test Data")
    
fig.tight_layout()
plt.show()

![image](https://user-images.githubusercontent.com/120009186/234347715-c0eb1739-97d5-48d9-8dd9-bfdd4b76508f.png)

- 이를 통해 데이터 분포에서 층화가 관측된 변수의 구간을 나눠 정확도를 측정 하였지만 이는 정확도를 오히려 낮췄다.

# 정규화
features = [col for col in train.columns if col != 'quality']
from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
train[features] = standardScaler.fit_transform(train[features])
test[features] = standardScaler.transform(test[features])

### 가장 정확도에 좋았던 단계는 이상치 제거 인데 Target 별 변수의 이상치들을 제거 해 줄때 였다.
train.drop('Id',axis=1,inplace=True)
test.drop('Id',axis=1,inplace=True)

- 이상치 제거 함수
def get_outlier2(df, column, weight=1.5):

    fraud = df[column]            
    quantile_25 = np.percentile(fraud.values, 25) # np.percentile
    quantile_75 = np.percentile(fraud.values, 75)
    
     
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight
   
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index.values

    return outlier_index
train.columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
       'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
       'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol',
       'quality']
target = 'quality'
features = [col for col in train.columns if col != target]
encoder = LabelEncoder()
train[target] = encoder.fit_transform(train[target])
outlier = set()
- target 별 이상치 제거
for i in range(0,6):
    for j in range(0,11):
        for k in get_outlier2(train.loc[train['quality']==i],features[j]):
            outlier.add(k)
train.drop(outlier,inplace=True)
train.reset_index(drop=True, inplace=True)
# 제거 확인
![image](https://user-images.githubusercontent.com/120009186/234349611-b213451f-fd63-45c8-8d71-017c75a024c5.png)

- 모델링
lgbm모델과 pca를 적용한 lgbm모델, catboost, xgboost모델을 비교 분석하였다.

![image](https://github.com/doubleleepark/kaggle_wine/assets/120009186/b58ec070-7f0a-466d-9483-3774cbe0bb56)

pca는 수치에 따라 8~9개를 활용

![image](https://github.com/doubleleepark/kaggle_wine/assets/120009186/3611d731-9337-41ac-93cc-2908feaefd9f)














