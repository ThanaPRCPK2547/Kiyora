import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

data = pd.read_csv('dataset_cleansing.csv')
data = data.drop('Timestamp', axis=1) 

# Rename columns
rename_map = {
    'อายุ': 'age',
    'โปรดเลือกรายได้ต่อเดือนของคุณ': 'monthly_income',
    'โปรดเลือกประเภทผิวของคุณ': 'skin_type',
    'คุณมีความกังวล/ปัญหาผิวในเรื่องใดบ้าง (เลือกได้หลายข้อ)': 'skin_concerns',
    'คุณเป็นสิวหรือไม่ เป็นสิวรุนแรงระดับใด': 'acne_level',
    "คุณปรึกษาหรือได้รับอิทธิพลจากใครในการเลือกสกินแคร์ 'สำหรับผิวหน้า' บ้าง (เลือกได้หลายคำตอบ)": 'skincare_influence',
    "คุณเลือกสกินแคร์ 'สำหรับผิวหน้า' อย่างไร (เลือกได้หลายข้อ)": 'skincare_selection_method',
    'คุณเลือกคลีนซิ่ง เช่น Cleansing water, cleansing balm, cleansing oil อย่างไร (เลือกได้หลายข้อ)': 'cleansing_selection_method',
    'คุณใช้คลีนซิ่งแบบน้ำ (Cleansing water) หรือไม่': 'uses_cleansing_water_raw',
    'ปัจจุบันคุณใช้คลีนซิ่งแบบใดบ้าง (หากใช้หลายแบบ เลือกได้หลายคำตอบ)': 'cleansing_types_used',
    'คุณใช้คลีนซิ่งแบบใดมากที่สุด (เลือกคำตอบเดียว)': 'cleansing_type_primary',
    'คุณใช้คลีนซิ่ง (Cleansing water) สูตรใด (หากใช้หลายแบบ เลือกได้หลายคำตอบ)': 'cleansing_water_formula',
    'ในการพิจารณาซื้อคลีนซิ่ง คุณพิจารณาคุณสมบัติใดบ้าง (5=มีผลต่อการพิจารณามากที่สุด) [เช็ดเมคอัพสะอาดหมดจด (Deep cleansing)]': 'factor_deep_cleansing',
    'ในการพิจารณาซื้อคลีนซิ่ง คุณพิจารณาคุณสมบัติใดบ้าง (5=มีผลต่อการพิจารณามากที่สุด) [ช่วยลดสิว/ไม่ก่อให้เกิดสิวใหม่ (Acne-prone skin friendly)]': 'factor_acne_friendly',
    'ในการพิจารณาซื้อคลีนซิ่ง คุณพิจารณาคุณสมบัติใดบ้าง (5=มีผลต่อการพิจารณามากที่สุด) [อ่อนโยนต่อผิวแพ้ง่าย (Sensitive skin friendly)]': 'factor_sensitive_friendly',
    'ในการพิจารณาซื้อคลีนซิ่ง คุณพิจารณาคุณสมบัติใดบ้าง (5=มีผลต่อการพิจารณามากที่สุด) [ผ่านการทดสอบทางการแพทย์ (Hypoallergenic)]': 'factor_hypoallergenic',
    'ในการพิจารณาซื้อคลีนซิ่ง คุณพิจารณาคุณสมบัติใดบ้าง (5=มีผลต่อการพิจารณามากที่สุด) [เช็ดแล้วชุ่มชื้น ผิวไม่แห้งตึง (Skin moisturized)]': 'factor_moisturizing',
    'ในการพิจารณาซื้อคลีนซิ่ง คุณพิจารณาคุณสมบัติใดบ้าง (5=มีผลต่อการพิจารณามากที่สุด) [ลดแรงเสียดสีของสำลีกับใบหน้า เช็ดแล้วไม่แสบผิว (Low friction formula)]': 'factor_low_friction',
    'ในการพิจารณาซื้อคลีนซิ่ง คุณพิจารณาคุณสมบัติใดบ้าง (5=มีผลต่อการพิจารณามากที่สุด) [มีสารบำรุงในตัว (Skin nourishment)]': 'factor_nourishment',
    'ในการพิจารณาซื้อคลีนซิ่ง คุณพิจารณาคุณสมบัติใดบ้าง (5=มีผลต่อการพิจารณามากที่สุด) [เช็ดรอบดวงตาได้ไม่แสบตา (Eye-friendly)]': 'factor_eye_friendly',
    'ในการพิจารณาซื้อคลีนซิ่ง คุณพิจารณาคุณสมบัติใดบ้าง (5=มีผลต่อการพิจารณามากที่สุด) [ช่วยลด/ควบคุมความมัน (Oil control)]': 'factor_oil_control',
    'ในการพิจารณาซื้อคลีนซิ่ง คุณพิจารณาคุณสมบัติใดบ้าง (5=มีผลต่อการพิจารณามากที่สุด) [ไม่มีสารก่อการแพ้ (เช่น ไม่มีแอลกอฮอล์, ไม่มีน้ำหอม, ไม่มีสี)]': 'factor_no_allergen',
    'ปัจจัยใดบ้างที่ส่งผลต่อการเปลี่ยนหรือทดลองคลีนซิ่งใหม่ (อื่นๆ โปรดพิมพ์ระบุเหตุผลสั้นๆ)': 'switch_factors',
    'ปัจจุบันคุณใช้คลีนซิ่งแบรนด์ใดอยู่บ้าง (เลือกได้หลายคำตอบ, เลือกอื่นๆ โปรดระบุ)': 'brands_used',
    'ปัจจุบันคุณใช้คลีนซิ่งแบรนด์ใดบ่อยที่สุด (เลือกเพียงคำตอบเดียว)': 'brand_primary',
    'เพศ': 'gender',
    'โปรดระบุอาชีพของคุณ (อื่นๆ โปรดระบุ)': 'occupation',
    'โปรดพิมพ์จังหวัดที่อยู่อาศัยของคุณ เช่น กทม. , ขอนแก่น, ชลบุรี': 'province',
}
data.rename(columns=rename_map, inplace=True)

print("2. Data Cleaning & Standardization...")
# Handle Missing factors
factor_cols = [
    'factor_deep_cleansing', 'factor_acne_friendly', 'factor_sensitive_friendly',
    'factor_hypoallergenic', 'factor_moisturizing', 'factor_low_friction',
    'factor_nourishment', 'factor_eye_friendly', 'factor_oil_control', 'factor_no_allergen'
]
for col in factor_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# Fill text NAs
cat_missing_cols = ['cleansing_types_used', 'cleansing_type_primary', 'cleansing_water_formula',
                    'switch_factors', 'brands_used', 'skin_concerns']
for col in cat_missing_cols:
    data[col] = data[col].fillna('None')

# Standardization: Province Grouping
data['province_group'] = data['province'].apply(
    lambda x: 'Bangkok_Metropolitan' if pd.notnull(x) and ('กทม' in str(x) or 'กรุงเทพ' in str(x) or 'นนทบุรี' in str(x) or 'ปทุมธานี' in str(x) or 'สมุทรปราการ' in str(x)) else 'Provincial'
)

# Standardization: Occupation Grouping
def group_occupation(x):
    x = str(x)
    if 'นักศึกษา' in x or 'นักเรียน' in x: return 'Student'
    elif 'บริษัท' in x or 'เอกชน' in x: return 'Private_Employee'
    elif 'ธุรกิจส่วนตัว' in x or 'ค้าขาย' in x or 'ฟรีแลนซ์' in x: return 'Business_Freelance'
    elif 'ข้าราชการ' in x or 'รัฐวิสาหกิจ' in x or 'ราชการ' in x: return 'Gov_Employee'
    else: return 'Other'
data['occupation_group'] = data['occupation'].apply(group_occupation)

print("3. Feature Engineering...")
# Encoding basic demographics
age_map = {'ต่ำกว่า 18 ปี': 1, '18-22 ปี': 2, '23-28 ปี': 3, '29-34 ปี': 4, '35 ปี ขึ้นไป': 5}
income_map = {'ต่ำกว่า 10,000 บาท': 1, '10,001 - 14,999 บาท': 2, '15,000 - 19,999 บาท': 3,
              '20,000 - 24,999 บาท': 4, '25,000 - 29,999 บาท': 5, '30,000 - 34,999 บาท': 6,
              '35,000 - 39,999 บาท': 7, '40,000 บาท ขึ้นไป': 8}
acne_map = {
    'ไม่มีสิวเลย': 0,
    'นานๆทีเป็นสิว เช่น มีสิวเฉพาะช่วงมีประจำเดือน, พักผ่อนน้อย': 1,
    'สิวเล็กน้อย (ส่วนใหญ่เป็นสิวอุดตัน เป็นสิวอักเสบ/หัวหนองไม่เกิน 10เม็ด)': 2,
    'สิวปานกลาง (เป็นสิวอักเสบ/หัวหนองมากกว่า10เม็ด)': 3,
    'สิวรุนแรง (มีสิวทุกประเภทร่วมกันเป็นกลุ่มก้อน อักเสบนาน และมีหนองไหล)': 4
}
data['age_encoded'] = data['age'].map(age_map)
data['monthly_income_encoded'] = data['monthly_income'].map(income_map)
data['acne_level_encoded'] = data['acne_level'].map(acne_map)

# Get Dummies for Gender and Cleansing Water
dummy_gender = pd.get_dummies(data['gender'])
if len(dummy_gender.columns) == 2:
    dummy_gender.columns = ['gender_female', 'gender_male']
data = pd.concat([data, dummy_gender], axis=1)

dummy_cw = pd.get_dummies(data['uses_cleansing_water_raw'])
if len(dummy_cw.columns) == 2:
    dummy_cw.columns = ['uses_cleansing_water', 'not_uses_cleansing_water']
data = pd.concat([data, dummy_cw], axis=1)

# Multiple Response One-Hot Encoding: Skin Concerns
skin_concerns_dummies = data['skin_concerns'].str.get_dummies(sep=', ')
skin_concerns_dummies.columns = [f"prob_{c}" for c in skin_concerns_dummies.columns]
data = pd.concat([data, skin_concerns_dummies], axis=1)

print("4. Target Variable Imputation...")
target_col = 'brand_primary'
# Make sure required cols exist for RF
feature_cols = ['gender_female', 'gender_male', 'age_encoded', 'monthly_income_encoded', 'acne_level_encoded']
# Ensure they have no NAs before RF
data[feature_cols] = data[feature_cols].fillna(0)

# If 'uses_cleansing_water' not defined properly, fallback logic
cw_col = 'uses_cleansing_water' if 'uses_cleansing_water' in data.columns else data.columns[data.columns.str.startswith('uses_')][0]

train_df = data[(data[cw_col] == True) & (data[target_col].notna()) & (data[target_col] != 'None')].copy()
pred_df  = data[(data[cw_col] == False) | (data[target_col].isna()) | (data[target_col] == 'None')].copy()

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_pred  = pred_df[feature_cols]

if len(pred_df) > 0 and len(train_df) > 0:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    data.loc[pred_df.index, target_col] = model.predict(X_pred)

print("5. Feature Selection (Mutual Information)...")
# Select numeric features for selection
numeric_features = factor_cols + ['age_encoded', 'monthly_income_encoded', 'acne_level_encoded']
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(data[target_col].astype(str))

mi_scores = mutual_info_classif(data[numeric_features].fillna(0), y_encoded, random_state=42)
mi_series = pd.Series(mi_scores, index=numeric_features).sort_values(ascending=False)

print("\n--- Feature Importance (Mutual Info vs brand_primary) ---")
print(mi_series.head(10))

# Encode Skin Type
le_skin = LabelEncoder()
data['skin_type_encoded'] = le_skin.fit_transform(data['skin_type'].astype(str))

# Save prepared dataset
data.to_csv('dataset_extended_prepared.csv', index=False)
