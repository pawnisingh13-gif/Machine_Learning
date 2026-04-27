# Customer Segmentation (Advanced)

import pandas as pd
from sklearn.cluster import KMeans

df = pd.DataFrame({
    'age':[20,25,30,35,40,45,50],
    'income':[20000,30000,40000,50000,60000,70000,80000],
    'spending':[20,30,40,60,70,80,90]
})

X = df[['age','income','spending']]

model = KMeans(n_clusters=3)
df['group'] = model.fit_predict(X)

print(df)
