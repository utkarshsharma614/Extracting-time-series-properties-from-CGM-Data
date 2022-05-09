import pandas as pd
import numpy as np
import pickle
import pickle_compat
pickle_compat.patch()
from scipy.fftpack import fft
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def fourierTransform(arg):
    fourierTransform = fft(arg)
    argLength = len(arg)

    rep = 2/400
    magnitude = []

    freq = np.linspace(0, argLength * rep, argLength)
    for mag in fourierTransform:
        magnitude.append(np.abs(mag))
    
    sortedMagnitude = sorted(magnitude)
    maxMagnitude = sortedMagnitude[(-2)]

    maxFrequency = freq.tolist()[magnitude.index(maxMagnitude)]

    return [maxMagnitude, maxFrequency]

def rms(arg):    
    rms = 0
    for p in range(0, len(arg) - 1):
        rms = rms + np.square(arg[p])

    return np.sqrt(rms / len(arg))

def randomness(arg):
    argLength = len(arg)
    randomness = 0

    if argLength <= 1:
        return 0
    else:
        value, count = np.unique(arg, return_counts = True)
        frac = count / argLength
        notFrac = np.count_nonzero(frac)
        if notFrac <= 1:
            return 0
        for idx in frac:
            randomness -= idx * np.log2(idx)
        return randomness


def absMean(arg):
    avgVal = 0

    for idx in range(0, len(arg) - 1):
        avgVal = avgVal + np.abs(arg[(idx + 1)] - arg[idx])

    return avgVal / len(arg)

# Extracting the required features

def featureExtraction(mealNomealVal):
    attrs = pd.DataFrame()

    for idx in range(0, mealNomealVal.shape[0]):
        val = mealNomealVal.iloc[idx, :].tolist()
        attrs = attrs.append({ 
         'Minimum Value': min(val), 
         'Maximum Value': max(val),
         'Mean of Absolute Values1': absMean(val[:13]), 
         'Mean of Absolute Values2': absMean(val[13:]),  
         'Root Mean Square': rms(val),
         'Randomness': randomness(val), 
         'Max FFT Magnitude1': fourierTransform(val[:13])[0], 
         'Max FFT Frequency1': fourierTransform(val[:13])[1], 
         'Max FFT Magnitude2': fourierTransform(val[13:])[0], 
         'Max FFT Frequency2': fourierTransform(val[13:])[1]},
          ignore_index = True)

    return attrs

def meal_nomeal_time(newT, tDiff):
    mealInstances = []

    t1 = newT[0:len(newT)-1]
    t2 = newT[1:len(newT)]

    temp = list(np.array(t1) - np.array(t2))
    Values = list(zip(t1, t2, temp))

    for value in Values:
        if value[2] < tDiff:
            mealInstances.append(value[0])

    return mealInstances


def meal_nomeal_data(mealT, start, end, flag, newGlucoseData):
    grid = []
    
    for newT in mealT:
        mealStartIdx= newGlucoseData[newGlucoseData['datetime'].between(newT + pd.DateOffset(hours = start), newT + pd.DateOffset(hours = end))]
        if mealStartIdx.shape[0] < 24:
            continue

        gValues = mealStartIdx['Sensor Glucose (mg/dL)'].to_numpy()
        avg = mealStartIdx['Sensor Glucose (mg/dL)'].mean()

        if flag:
            cnt = 30 - len(gValues)
            if cnt > 0:
                for _ in range(cnt):
                    gValues = np.append(gValues, avg)
            grid.append(gValues[0:30])
        else:
            grid.append(gValues[0:24])

    return pd.DataFrame(data = grid)

# Processing the data
def dataWrangling(insulinData, glucoseData):
    mealData = pd.DataFrame()
    noMealData = pd.DataFrame()

    insulinData = insulinData[::-1]
    glucoseData = glucoseData[::-1]
    glucoseData['Sensor Glucose (mg/dL)'] = glucoseData['Sensor Glucose (mg/dL)'].interpolate(method = 'linear',limit_direction = 'both')
    
    
    insulinData['datetime'] = pd.to_datetime(insulinData["Date"].astype(str) + " " + insulinData["Time"].astype(str))
    glucoseData['datetime'] = pd.to_datetime(glucoseData["Date"].astype(str) + " " + glucoseData["Time"].astype(str))
    
    newInsulinData = insulinData[['datetime','BWZ Carb Input (grams)']]
    newGlucoseData = glucoseData[['datetime','Sensor Glucose (mg/dL)']]
    
    newInsulinData = newInsulinData[(newInsulinData['BWZ Carb Input (grams)'].notna()) & (newInsulinData['BWZ Carb Input (grams)']>0) ]
    
    newT = list(newInsulinData['datetime'])
    
    mealT = []
    noMealT = []

    mealT = meal_nomeal_time(newT, pd.Timedelta('0 days 120 min'))
    noMealT = meal_nomeal_time(newT, pd.Timedelta('0 days 240 min'))
    
    mealData = meal_nomeal_data(mealT, -0.5, 2, True, newGlucoseData)
    noMealData = meal_nomeal_data(noMealT, 2, 4, False, newGlucoseData)

    mealFeatures = featureExtraction(mealData)
    noMealFeatures = featureExtraction(noMealData)
 
    mealStd = StandardScaler().fit_transform(mealFeatures)
    noMealStd = StandardScaler().fit_transform(noMealFeatures)
    
    pca = PCA(n_components = 5)
    pca.fit(mealStd)
         
    mealPCA = pd.DataFrame(pca.fit_transform(mealStd))
    noMealPCA = pd.DataFrame(pca.fit_transform(noMealStd))
    
    mealPCA['class'] = 1
    noMealPCA['class'] = 0
    
    arr = mealPCA.append(noMealPCA)
    arr.index = [idx for idx in range(arr.shape[0])]
    return arr


# Using SVM Model
if __name__ == '__main__':
    insulinData1 = pd.read_csv("Insulin_patient2.csv",low_memory=False)
    glucoseData1 = pd.read_csv("CGM_patient2.csv",low_memory=False)
    insulinData2 = pd.read_csv("InsulinData.csv",low_memory=False)
    glucoseData2 = pd.read_csv("CGMData.csv",low_memory=False)

    insulinData = pd.concat([insulinData1, insulinData2])
    glucoseData = pd.concat([glucoseData1, glucoseData2])
    
    arr = dataWrangling(insulinData, glucoseData)

    p = arr.iloc[:, :-1]
    q = arr.iloc[:, -1]
    
    model = SVC(kernel = 'linear', C = 1, gamma = 0.1)

    folds = KFold(5, True, 1)

    for train, test in folds.split(p, q):
        pTrain, pTest = p.iloc[train], p.iloc[test]
        qTrain, qTest = q.iloc[train], q.iloc[test]
        
        model.fit(pTrain, qTrain)

    with open('model.pkl', 'wb') as (file):
        pickle.dump(model, file)
