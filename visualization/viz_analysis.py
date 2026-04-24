import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Thai font support
plt.rcParams['font.family'] = 'Tahoma'

# โหลดข้อมูล
data = pd.read_csv('dataset_extended_prepared.csv')

# Part 1: Demographic Profile
# เพศ (Gender)
plt.figure(figsize=(8, 5))
if 'gender_female' in data.columns and 'gender_male' in data.columns:
    gender_counts = {'Female': data['gender_female'].sum(), 'Male': data['gender_male'].sum()}
    x_keys = list(gender_counts.keys())
    sns.barplot(x=x_keys, y=list(gender_counts.values()), hue=x_keys, palette='pastel', legend=False)
    plt.title('Distribution of Gender (การกระจายตัวของเพศ)')
    plt.ylabel('Count (จำนวน)')
    plt.show()

# อายุ (Age)
plt.figure(figsize=(10, 5))
age_labels = ['< 18', '18-22', '23-28', '29-34', '35+']
age_counts = data['age_encoded'].value_counts().sort_index()
sns.barplot(x=age_labels, y=age_counts.values, hue=age_labels, palette='Blues_d', legend=False)
plt.title('Distribution of Age Groups (การกระจายตัวของช่วงอายุ)')
plt.ylabel('Count (จำนวน)')
plt.show()

# อาชีพ (Occupation)
plt.figure(figsize=(10, 5))
sns.countplot(data=data, y='occupation_group', order=data['occupation_group'].value_counts().index, hue='occupation_group', palette='Set2', legend=False)
plt.title('Distribution of Occupations (การกระจายตัวของกลุ่มอาชีพ)')
plt.xlabel('Count (จำนวน)')
plt.ylabel('Occupation Group (กลุ่มอาชีพ)')
plt.show()

# ประเภทผิว (Skin Type)
plt.figure(figsize=(8, 8))
skin_counts = data['skin_type'].value_counts()
plt.pie(skin_counts, labels=skin_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set3'))
plt.title('Proportion of Skin Types (สัดส่วนของประเภทผิว)')
plt.show()

# จังหวัด (Province)
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='province_group', hue='province_group', palette='muted', legend=False)
plt.title('Distribution of Provinces (จังหวัด: กทม.และปริมณฑล vs ต่างจังหวัด)')
plt.ylabel('Count (จำนวน)')
plt.show()

# Part 2: Distribution Analysis

# Top 10 แบรนด์หลัก
plt.figure(figsize=(12, 6))
top_brands = data['brand_primary'].value_counts().head(10)
sns.barplot(x=top_brands.values, y=top_brands.index, hue=top_brands.index, palette='viridis', legend=False)
plt.title('Top 10 Primary Brands Used (10 อันดับแบรนด์หลักที่ใช้งานมากที่สุด)')
plt.xlabel('Number of Users (จำนวนผู้ใช้งาน)')
plt.ylabel('Brand (แบรนด์)')
plt.show()

# คะแนนเฉลี่ยปัจจัยคลีนซิ่ง
factor_cols = [
    'factor_deep_cleansing', 'factor_acne_friendly', 'factor_sensitive_friendly',
    'factor_hypoallergenic', 'factor_moisturizing', 'factor_low_friction',
    'factor_nourishment', 'factor_eye_friendly', 'factor_oil_control', 'factor_no_allergen'
]
factor_means = data[factor_cols].replace(0, pd.NA).mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
factor_names = [col.replace('factor_', '') for col in factor_means.index]
sns.barplot(x=factor_means.values, y=factor_names, hue=factor_names, palette='magma', legend=False)
plt.title('Average Importance Score for Cleansing Properties (คะแนนความสำคัญเฉลี่ยของคุณสมบัติคลีนซิ่ง)')
plt.xlabel('Average Score (คะแนนเฉลี่ย 1-5)')
plt.ylabel('Factor (ปัจจัย)')
plt.show()

# Part 3: Correlation Heatmap

# Correlation ระหว่างปัจจัย
plt.figure(figsize=(10, 8))
corr_factors = data[factor_cols].corr()
sns.heatmap(corr_factors, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix of Cleansing Factors (เมทริกซ์ความสัมพันธ์ของปัจจัยการเลือกคลีนซิ่ง)')
plt.show()

# Correlation กับ Target (Brand Primary)
plt.figure(figsize=(10, 8))
features_for_heatmap = ['age_encoded', 'monthly_income_encoded', 'acne_level_encoded', 'skin_type_encoded'] + factor_cols
data['brand_encoded'] = LabelEncoder().fit_transform(data['brand_primary'].astype(str))
cols_to_correlate = features_for_heatmap + ['brand_encoded']
corr_target = data[cols_to_correlate].corr()
brand_corr = corr_target[['brand_encoded']].drop('brand_encoded').sort_values(by='brand_encoded', ascending=False)
sns.heatmap(brand_corr, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1)
plt.title('Correlation of Features with Target (ความสัมพันธ์ของฟีเจอร์กับแบรนด์หลักที่ใช้งาน)')
plt.show()
