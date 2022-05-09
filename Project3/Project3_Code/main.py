import math
import pandas as pd
import datetime
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def entropyDBScanRow(r, dbScanConfMat):
    enp = 0
    res = 0
    for i in range(len(dbScanConfMat.columns)):
        res = res + r[i]

    for j in range(len(dbScanConfMat.columns)):
        if (r[j] == 0):
            continue
        enp = enp + r[j] / res * math.log(r[j] / res, 2)
    return -enp

def entropyOfRow(r, confMatData):
    enp = 0
    res = 0
    for i in range(len(confMatData.columns)):
        res += r[i]

    for j in range(len(confMatData.columns)):
        if (r[j] == 0):
            continue
        enp +=  r[j] / res * math.log(r[j] / res, 2)
    return -enp

def headerBin(idx):
    if (idx <= 23):
        return np.floor(0)
    elif (idx <= 43):
        return np.floor(1)
    elif (idx <= 63):
        return np.floor(2)
    elif (idx <= 83):
        return np.floor(3)
    elif (idx <= 103):
        return np.floor(4)
    else:
        return np.floor(5)

def findCGMMeanCentroid(row):
    return cgmMeanCentroidObj[row['cluster']]

def findCarbCentroid(row):
    return carbCentroidObj[row['cluster']]

cgmData = pd.read_csv('CGMData.csv', sep=',', low_memory=False)
cgmData['dateTime'] = pd.to_datetime(cgmData['Date'] + ' ' + cgmData['Time'])
cgmData = cgmData.sort_values(by='dateTime', ascending=True)

insulinData = pd.read_csv('InsulinData.csv', sep = ',', low_memory = False)
insulinData['dateTime'] = pd.to_datetime(insulinData['Date'] + ' ' + insulinData['Time'])
insulinData = insulinData.sort_values(by = 'dateTime', ascending = True)

insulinData['Idx New'] = range(0, 0 + len(insulinData))
mealDuration = insulinData.loc[insulinData['BWZ Carb Input (grams)'] > 0][['Idx New', 'Date', 'Time', 'BWZ Carb Input (grams)', 'dateTime']]
mealDuration['diff'] = mealDuration['dateTime'].diff(periods=1)
mealDuration['moveUP'] = mealDuration['diff'].shift(-1)

mealDuration = mealDuration.loc[(mealDuration['moveUP'] > datetime.timedelta(minutes=120)) | (pd.isnull(mealDuration['moveUP']))]

cgmDataMealdata = pd.DataFrame()
cgmDataMealdata['Idx New'] = ""
for i in range(len(mealDuration)):
    afterMealDuration = mealDuration['dateTime'].iloc[i] + datetime.timedelta(minutes=120)
    beforeMealDuration = mealDuration['dateTime'].iloc[i] - datetime.timedelta(minutes=30)

    cgmdata_durationMeal = cgmData.loc[(cgmData['dateTime'] >= beforeMealDuration) & (cgmData['dateTime'] < afterMealDuration)]
    res = []
    index = 0
    index = mealDuration['Idx New'].iloc[i]
    for j in range(len(cgmdata_durationMeal)):
        res.append(cgmdata_durationMeal['Sensor Glucose (mg/dL)'].iloc[j])
    cgmDataMealdata = cgmDataMealdata.append(pd.Series(res), ignore_index=True)
    cgmDataMealdata.iloc[i, cgmDataMealdata.columns.get_loc('Idx New')] = index
cgmDataMealdata['Idx New'] = cgmDataMealdata['Idx New'].astype(int)

cgmMealDataIdx = pd.DataFrame()
cgmMealDataIdx['Idx New'] = cgmDataMealdata['Idx New']
cgmDataMealdata = cgmDataMealdata.drop(columns='Idx New')

# Performing the Interpolation using the below code

cntRows = cgmDataMealdata.shape[0]
cntColumns = cgmDataMealdata.shape[1]

cgmDataMealdata.dropna(axis = 1, how = 'all', thresh = cntRows / 4, subset = None, inplace = True)
cgmDataMealdata.dropna(axis = 0, how = 'all', thresh = cntColumns / 4, subset = None, inplace = True)

cgmDataMealdata.interpolate(axis = 0, method = 'linear', limit_direction = 'forward', inplace = True)
cgmDataMealdata.bfill(axis=1, inplace=True)

cgm_NoMealdata_index = cgmDataMealdata.copy()
cgm_mean = cgmDataMealdata.copy()

cgmMealData = pd.merge(cgmDataMealdata, cgmMealDataIdx, left_index = True, right_index = True)
cgmMealData['mean CGM data'] = cgm_NoMealdata_index.mean(axis=1)
cgmMealData['max-start_over_start'] = cgm_NoMealdata_index.max(axis=1) / cgm_NoMealdata_index[0]

mealAmount = mealDuration[['BWZ Carb Input (grams)', 'Idx New']]
mealAmount = mealAmount.rename(columns={'BWZ Carb Input (grams)': 'Meal Amount'})

mealAmtHeader = pd.DataFrame()
mealAmtHeader['Bin Label'] = mealAmount.apply(lambda row: headerBin(row['Meal Amount']).astype(np.int64), axis = 1)
mealAmtHeader['Idx New'] = mealAmount['Idx New']

mealDataAmount = cgmMealData.merge(mealAmtHeader, how='inner', on=['Idx New'])

mealCarbIntakeDuration = pd.DataFrame()
mealCarbIntakeDuration = mealDuration[['BWZ Carb Input (grams)', 'Idx New']]
mealDataAmount = mealDataAmount.merge(mealCarbIntakeDuration, how='inner', on=['Idx New'])
mealDataAmount = mealDataAmount.drop(columns='Idx New')

extractCarbFeature = pd.DataFrame()
extractCarbFeature = mealDataAmount[['BWZ Carb Input (grams)', 'mean CGM data']]

kMeansVal = extractCarbFeature.copy()
kMeansVal = kMeansVal.values.astype('float32', copy = False)
kmeans_data = StandardScaler().fit(kMeansVal)
Feature_extraction_scaler = kmeans_data.transform(kMeansVal)

kMeansExtent = range(1, 17)
sse = []
for k in kMeansExtent:
    kMeansTestFeatures = KMeans(n_clusters=k,random_state=45)
    kMeansTestFeatures.fit(Feature_extraction_scaler)
    sse.append(kMeansTestFeatures.inertia_)

kMeansAns = KMeans(n_clusters=17,random_state=45)
kMeansYValPrediction = kMeansAns.fit_predict(Feature_extraction_scaler)
kMeansSSE = kMeansAns.inertia_

extractCarbFeature['cluster'] = kMeansYValPrediction
extractCarbFeature.head()

kMeansAns.cluster_centers_

ground_truthdata_array = mealDataAmount["Bin Label"].tolist()

binClusterData = pd.DataFrame({'ground_true_arr': ground_truthdata_array, 'kmeans_labels': list(kMeansYValPrediction)}, columns=['ground_true_arr', 'kmeans_labels'])

confMatData = pd.pivot_table(binClusterData, index='kmeans_labels', columns='ground_true_arr', aggfunc = len)
confMatData.fillna(value=0, inplace=True)
confMatData = confMatData.reset_index()
confMatData = confMatData.drop(columns=['kmeans_labels'])

confMatCpy = confMatData.copy()

confMatCpy['Total'] = confMatData.sum(axis=1)
confMatCpy['Row_entropy'] = confMatData.apply(lambda row: entropyOfRow(row,confMatData), axis=1)
sumData = confMatCpy['Total'].sum()
confMatCpy['entropy_prob'] = confMatCpy['Total'] / sumData * confMatCpy['Row_entropy']
kMeansEnp = confMatCpy['entropy_prob'].sum()

confMatCpy['Max_val'] = confMatData.max(axis = 1)
kMeansPurityData = confMatCpy['Max_val'].sum() / sumData

dbScanAttrs = extractCarbFeature.copy()[['BWZ Carb Input (grams)', 'mean CGM data']]
dbScanFeatureArray = dbScanAttrs.values.astype('float32', copy = False)
dbScanDataScale = StandardScaler().fit(dbScanFeatureArray)
dbScanFeatureArray = dbScanDataScale.transform(dbScanFeatureArray)

model = DBSCAN(eps = 0.20, min_samples = 5).fit(dbScanFeatureArray)

extractCarbFeature['cluster'] = model.labels_

clusters = Counter(model.labels_)

clusterBinDBScan = pd.DataFrame({'ground_true_arr': ground_truthdata_array, 'dbscan_labels': list(model.labels_)}, columns = ['ground_true_arr', 'dbscan_labels'])
dbScanConfMat = pd.pivot_table(clusterBinDBScan, index = 'ground_true_arr', columns = 'dbscan_labels', aggfunc = len)
dbScanConfMat.fillna(value=0, inplace = True)
dbScanConfMat = dbScanConfMat.reset_index()
dbScanConfMat = dbScanConfMat.drop(columns=['ground_true_arr'])
dbScanConfMat = dbScanConfMat.drop(columns=[-1])
dbScanConfMatCpy = dbScanConfMat.copy()
dbScanConfMatCpy['Total'] = dbScanConfMat.sum(axis = 1)
dbScanConfMatCpy['Row_entropy'] = dbScanConfMat.apply(lambda row: entropyDBScanRow(row,dbScanConfMat), axis=1)

sumData = dbScanConfMatCpy['Total'].sum()
dbScanConfMatCpy['entropy_prob'] = dbScanConfMatCpy['Total'] / sumData * dbScanConfMatCpy['Row_entropy']
dbScanEntropy = dbScanConfMatCpy['entropy_prob'].sum()
dbScanConfMatCpy['Max_val'] = dbScanConfMat.max(axis = 1)
DBSCAN_purity = dbScanConfMatCpy['Max_val'].sum() / sumData

extractCarbFeature = extractCarbFeature.loc[extractCarbFeature['cluster'] != -1]

dbScanFeatureCentroid = extractCarbFeature.copy()
carbCentroidObj = {}
cgmMeanCentroidObj = {}
squareOfError = {}
DBSCAN_SSE = 0

for i in range(len(dbScanConfMat.columns)):
    clusterReg = extractCarbFeature.loc[extractCarbFeature['cluster'] == i]
    carbInputCentroid = clusterReg['BWZ Carb Input (grams)'].mean()
    cgmMeanCentroid = clusterReg['mean CGM data'].mean()
    carbCentroidObj[i] = carbInputCentroid
    cgmMeanCentroidObj[i] = cgmMeanCentroid

dbScanFeatureCentroid['carbInputCentroid'] = extractCarbFeature.apply(lambda row: findCarbCentroid(row), axis = 1)
dbScanFeatureCentroid['cgmMeanCentroid'] = extractCarbFeature.apply(lambda row: findCGMMeanCentroid(row), axis = 1)
dbScanFeatureCentroid['centroidDiff'] = 0

for i in range(len(dbScanFeatureCentroid)):
    dbScanFeatureCentroid['centroidDiff'].iloc[i] = math.pow(dbScanFeatureCentroid['BWZ Carb Input (grams)'].iloc[i] - dbScanFeatureCentroid['carbInputCentroid'].iloc[i], 2) + math.pow(dbScanFeatureCentroid['mean CGM data'].iloc[i] - dbScanFeatureCentroid['cgmMeanCentroid'].iloc[i], 2)
    
for i in range(len(dbScanConfMat.columns)):
    squareOfError[i] = dbScanFeatureCentroid.loc[dbScanFeatureCentroid['cluster'] == i]['centroidDiff'].sum()

for i in squareOfError:
    DBSCAN_SSE = DBSCAN_SSE + squareOfError[i]

KMeans_DBSCAN = [kMeansSSE, DBSCAN_SSE, kMeansEnp, dbScanEntropy, kMeansPurityData, DBSCAN_purity]
df = pd.DataFrame(KMeans_DBSCAN).T
df.to_csv('Results.csv', header = False, index = False)