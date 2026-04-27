import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("dataset_extended_prepared.csv")

cluster_features = [
    'age_encoded',
    'monthly_income_encoded',
    'acne_level_encoded',
    'skin_type_encoded',
    'factor_deep_cleansing',
    'factor_acne_friendly',
    'factor_sensitive_friendly',
    'factor_hypoallergenic',
    'factor_moisturizing',
    'factor_low_friction',
    'factor_nourishment',
    'factor_eye_friendly',
    'factor_oil_control',
    'factor_no_allergen'
]

X = data[cluster_features].fillna(0)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original Shape:", X.shape)
print("Scaled Shape:", X_scaled.shape)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

inertia = []
sil_scores = []

K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Elbow Plot
plt.figure(figsize=(8,4))
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Silhouette Plot
plt.figure(figsize=(8,4))
plt.plot(K_range, sil_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Score")
plt.show()

print(list(zip(K_range, sil_scores)))

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_scaled)

print(data['cluster'].value_counts())


cluster_profile = data.groupby('cluster')[cluster_features].mean().round(2)
print(cluster_profile)

#เเบ่ง กลุ่มลูกค้าเป็น 4 กลุ่มตาม cluster ที่ได้จาก KMeans
#Cluster 0 = Young Passive Buyers (วัยรุ่นงบน้อย ซื้อแบบง่าย ๆ)
#Cluster 1 = Young Active Skincare Seekers(คนรุ่นใหม่ สนใจ skincare จริง)
#Cluster 2 = Premium Rational Buyers(ซื้อเพราะคุณภาพ ไม่ใช่ราคาถูก)
#Cluster 3 = Mature Habit Buyers(อายุมาก ซื้อจากความเคยชิน)

#ดูข้อมูลเเต่ละ brand/cluster
brand_cluster = pd.crosstab(
    data['cluster'],
    data['brand_primary'],
    normalize='index'
).round(3)

print(brand_cluster.to_string())