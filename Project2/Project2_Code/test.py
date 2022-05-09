import pandas as pd
import pickle
import pickle_compat
pickle_compat.patch()
from sklearn.decomposition import PCA
from train import featureExtraction
from sklearn.preprocessing import StandardScaler


with open("model.pkl", 'rb') as file:
        testingFrame = pd.read_csv('test.csv', header = None)
        model = pickle.load(file) 
        
extractedFeatures = featureExtraction(testingFrame)
scaler = StandardScaler().fit_transform(extractedFeatures)    

output = PCA(n_components = 5)
predictedValues = model.predict(output.fit_transform(scaler))

pd.DataFrame(predictedValues).to_csv("Results.csv", header = None, index = False)