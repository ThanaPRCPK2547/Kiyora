import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('dataset_cleansing.csv')
print(data.head())
print(data.info())
print(data.isnull().sum())

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

# จัดการ Missing Values จำนวน 47 rows (กลุ่มคนที่ไม่ได้ใช้คลีนซิ่ง)
# กลุ่มคอลัมน์ให้คะแนน ปัจจัยต่างๆ (factor_...) เติม 0
factor_cols = [
    'factor_deep_cleansing', 'factor_acne_friendly', 'factor_sensitive_friendly',
    'factor_hypoallergenic', 'factor_moisturizing', 'factor_low_friction',
    'factor_nourishment', 'factor_eye_friendly', 'factor_oil_control'
]
for col in factor_cols:
    data[col].fillna(0, inplace=True)

# กลุ่มคอลัมน์ข้อความพฤติกรรมการใช้ ให้เติมเป็น 0
cat_missing_cols = [
    'cleansing_types_used', 'cleansing_type_primary', 'cleansing_water_formula',
    'switch_factors', 'brands_used'
]
for col in cat_missing_cols:
    data[col].fillna(0, inplace=True)

# ทำ prep data
def data_preparation(data):
    dummy = pd.get_dummies(data['gender'])
    dummy.columns = ['gender_female', 'gender_male']
    data = pd.concat([data, dummy], axis='columns')
    data.drop(['gender'], axis=1, inplace=True)
    dummy_2 = pd.get_dummies(data['uses_cleansing_water_raw'])
    dummy_2.columns = ['uses_cleansing_water', 'not_uses_cleansing_water']
    data = pd.concat([data, dummy_2], axis='columns')
    data.drop(['uses_cleansing_water_raw'], axis=1, inplace=True)
    return data

data = data_preparation(data)
data = data.drop(['Timestamp', 'province', 'occupation', 'factor_no_allergen'], axis=1)

print(data.head(5))

#กำหนดคอลัมน์ที่ใช้เป็นฟีเจอร์และเป้าหมาย
feature_cols = ['gender_male', 'gender_female', 'age', 'monthly_income', 'skin_type', 'acne_level']
#กำหนดคอลัมน์เป้าหมาย
target_col = 'brand_primary'

train_df = data[(data['uses_cleansing_water'] == True) & (data[target_col].notna())].copy()
pred_df  = data[(data['not_uses_cleansing_water'] == True) & (data[target_col].isna())].copy()

all_x = pd.concat([train_df[feature_cols], pred_df[feature_cols]], axis=0)
all_x = pd.get_dummies(all_x)

X_train = all_x.iloc[:len(train_df)]
X_pred  = all_x.iloc[len(train_df):]
y_train = train_df[target_col]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
data.loc[pred_df.index, target_col] = model.predict(X_pred)

print(data[[target_col]].isnull().sum())
print(data.loc[pred_df.index, target_col].head())
print("\nFinal columns:")
print(data.columns.tolist())
print(data.head())
print(data.info())

# ทำ Ordinal Encoding (แปลง Text เป็นตัวเลขตามลำดับ) เฉพาะคอลัมน์ใน feature_cols
age_map = {
    'ต่ำกว่า 18 ปี': 1, '18-22 ปี': 2, '23-28 ปี': 3, '29-34 ปี': 4, '35 ปี ขึ้นไป': 5
}
income_map = {
    'ต่ำกว่า 10,000 บาท': 1, '10,001 - 14,999 บาท': 2, '15,000 - 19,999 บาท': 3,
    '20,000 - 24,999 บาท': 4, '25,000 - 29,999 บาท': 5, '30,000 - 34,999 บาท': 6,
    '35,000 - 39,999 บาท': 7, '40,000 บาท ขึ้นไป': 8
}
acne_map = {
    'ไม่มีสิวเลย': 0,
    'นานๆทีเป็นสิว เช่น มีสิวเฉพาะช่วงมีประจำเดือน, พักผ่อนน้อย': 1,
    'สิวเล็กน้อย (ส่วนใหญ่เป็นสิวอุดตัน เป็นสิวอักเสบ/หัวหนองไม่เกิน 10เม็ด)': 2,
    'สิวปานกลาง (เป็นสิวอักเสบ/หัวหนองมากกว่า10เม็ด)': 3,
    'สิวรุนแรง (มีสิวทุกประเภทร่วมกันเป็นกลุ่มก้อน อักเสบนาน และมีหนองไหล)': 4
}

data['age'] = data['age'].map(age_map)
data['monthly_income'] = data['monthly_income'].map(income_map)
data['acne_level'] = data['acne_level'].map(acne_map)

# ทำ Label Encoding สำหรับ Categorical variables ใน feature_cols
label_cols = ['skin_type']
le = LabelEncoder()
for col in label_cols:
    if col in data.columns:
        # เพื่อความปลอดภัย จัดการให้เป็นประเภท string ทั้งหมดก่อน
        data[col] = data[col].astype(str)
        data[col] = le.fit_transform(data[col])

# export data set
print(data.info())
print(data.isnull().sum())
print(data.to_csv('dataset_prepared.csv', index=False))
