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

#หลังจากดูข้อมูลสรุปได้ว่าลูกค้ากลุ่ม cluster 0 ใช้ Kiyora เป็นหลัก

# =====================================================
# MODEL 2 : KIYORA USER SEGMENTATION
# =====================================================


# Filter เฉพาะคนที่ใช้ Kiyora


kiyora_df = data[
    data["brand_primary"].astype(str).str.contains("Kiyora", case=False, na=False)
].copy()

print("Kiyora Users:", kiyora_df.shape[0])



#  เลือก Features ที่สะท้อนเหตุผลการเลือกซื้อ


cluster_features = [

    # Skin condition
    "skin_type_encoded",
    "acne_level_encoded",

    # Product drivers
    "factor_deep_cleansing",
    "factor_acne_friendly",
    "factor_sensitive_friendly",
    "factor_hypoallergenic",
    "factor_moisturizing",
    "factor_low_friction",
    "factor_nourishment",
    "factor_eye_friendly",
    "factor_oil_control",
    "factor_no_allergen"
]

# ถ้ามี prob_ columns (skin concerns) ให้เพิ่มอัตโนมัติ
prob_cols = [col for col in kiyora_df.columns if col.startswith("prob_")]
cluster_features.extend(prob_cols)

print("Total Features:", len(cluster_features))



# เตรียมข้อมูล


X = kiyora_df[cluster_features].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# : หา k ที่เหมาะสม


inertia = []
sil_scores = []

K_range = range(2, 7)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")

plt.subplot(1,2,2)
plt.plot(K_range, sil_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Score")

plt.tight_layout()
plt.show()


# ทํา kทำฟืห


best_k = 3

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kiyora_df["cluster"] = kmeans.fit_predict(X_scaled)



# ดูจำนวนคนแต่ละกลุ่ม


print("\nCluster Counts")
print(kiyora_df["cluster"].value_counts())



# Profile แต่ละ Cluster


cluster_profile = kiyora_df.groupby("cluster")[cluster_features].mean().round(2)

pd.set_option("display.max_columns", None)
print("\nCluster Profile")
print(cluster_profile.to_string())



kiyora_df.to_csv("kiyora_clustered.csv", index=False)
cluster_profile.to_csv("kiyora_cluster_profile.csv")

print("\nFiles Saved:")
print("- kiyora_clustered.csv")
print("- kiyora_cluster_profile.csv")