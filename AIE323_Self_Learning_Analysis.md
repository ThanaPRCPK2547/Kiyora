# AIE323 Self-Learning Assignment - การวิเคราะห์สิ่งที่ต้องทำเพิ่ม

## ข้อมูลการส่งงาน
- **วันที่ส่ง:** 27 เมษายน 2569 (ก่อนเที่ยงคืน)
- **ไฟล์ที่ต้องส่ง:** Code file, CSV file, Slide presentation
- **วันพรีเซนต์:**
  - Section 338B: 28 เมษายน 2569
  - Section 338A: 30 เมษายน 2569

---

## สิ่งที่ทำเสร็จแล้ว (ใน data_prep.py)

### 1. Data Loading & Basic Exploration
- [x] อ่านไฟล์ dataset_cleansing.csv
- [x] แสดงข้อมูลเบื้องต้น (head, info, isnull)

### 2. Data Cleaning
- [x] Rename columns (เปลี่ยนชื่อคอลัมน์เป็นภาษาอังกฤษ)
- [x] Handle missing values บางส่วน

### 3. Feature Engineering
- [x] Encoding: gender (get_dummies)
- [x] Encoding: uses_cleansing_water_raw (get_dummies)
- [x] Drop columns: Timestamp, province, occupation, factor_no_allergen

### 4. Model Building (มีอยู่แล้ว)
- [x] RandomForestClassifier สำหรับ predict brand_primary

---

## สิ่งที่ต้องทำเพิ่ม (ตามข้อกำหนด Assignment)

### 1. Target Variable Identification & Labeling ⚠️ **ต้องทำเพิ่ม**

#### 1.1 Define Target (ต้องระบุให้ชัดเจน)
```
❌ ปัจจุบัน: ใช้ 'brand_primary' เป็น target โดยไม่ได้ให้เหตุผล
✅ ต้องทำ: ระบุว่าทำไมถึงเลือก target นี้ และต้องการศึกษาอะไร
```

**แนะนำ:** สร้างเอกสารอธิบายว่า
- Target variable คืออะไร (brand_primary - แบรนด์คลีนซิ่งที่ใช้บ่อยที่สุด)
- เหตุผลที่เลือก: ต้องการทำนายว่าผู้บริโภคจะเลือกใช้แบรนด์คลีนซิ่งใดบ่อยที่สุด
- ประโยชน์: ช่วยในการวางแผนการตลาดและพัฒนาผลิตภัณฑ์

#### 1.2 Data Labeling (ตรวจสอบว่าจำเป็นหรือไม่)
```
✅ ปัจจุบัน: มี brand_primary อยู่แล้วในข้อมูล (ไม่จำเป็นต้องสร้าง label ใหม่)
⚠️ ต้องตรวจสอบ: มี row ที่ brand_primary เป็น NA หรือไม่
```

---

### 2. Data Cleaning ⚠️ **ต้องทำเพิ่ม**

#### 2.1 Handle Incomplete Responses (ต้องทำ)
```python
# ต้องเพิ่มโค้ดตรวจสอบและจัดการกับ:
# - แถวที่ตอบไม่ครบ (มี NaN จำนวนมาก)
# - แถวที่ข้อมูลสำคัญหายไป
```

**สิ่งที่ต้องทำ:**
- นับจำนวน NaN ต่อ row
- ตัด row ที่มี NaN เกินเกณฑ์ (เช่น เกิน 50% ของ columns)
- หรือ impute ค่าที่หายไป

#### 2.2 Standardization (ต้องทำ)
```
❌ ยังไม่ได้ทำ: การปรับคำตอบปลายเปิดให้เป็นหมวดหมู่
```

**Columns ที่ต้องตรวจสอบ:**
| Column | ปัจจุบัน | ต้องทำ |
|--------|----------|--------|
| occupation | มีค่าหลากหลายมาก | จัดกลุ่ม (พนักงานบริษัท, ธุรกิจส่วนตัว, นักเรียน/นักศึกษา, อื่นๆ) |
| province | มีชื่อจังหวัดหลากหลาย | จัดกลุ่ม (กทม., ปริมณฑล, ต่างจังหวัด) |
| skin_concerns | มีหลายค่ารวมกัน | แยกเป็น individual features |
| skincare_selection_method | มีหลายค่ารวมกัน | แยกเป็น individual features |
| cleansing_types_used | มีหลายค่ารวมกัน | แยกเป็น individual features |
| switch_factors | มีหลายค่ารวมกัน | แยกเป็น individual features |
| brands_used | มีหลายค่ารวมกัน | แยกเป็น individual features |

#### 2.3 Logic Check (ต้องทำ)
```
❌ ยังไม่ได้ทำ: ตรวจสอบความสมเหตุสมผล
```

**สิ่งที่ต้องตรวจสอบ:**
- เพศชาย แต่ตอบคำถามเกี่ยวกับประจำเดือน?
- อายุ < 18 แต่รายได้ 40,000+?
- ผิวแห้ง แต่ใช้สูตรคุมมัน?
- ไม่เป็นสิวเลย แต่เลือก factor_acne_friendly = 5?

---

### 3. Feature Selection ⚠️ **ต้องทำเพิ่ม**

#### 3.1 Feature Selection ด้วยค่าสถิติ (ต้องทำ)
```python
# ต้องเพิ่ม:
# - Correlation analysis
# - Chi-square test (สำหรับ categorical)
# - ANOVA F-test
# - Feature importance จากโมเดล
```

**สิ่งที่ต้องทำ:**
- สร้าง correlation matrix
- เลือก features ที่มี correlation กับ target สูง
- ตัด features ที่ไม่มีความสัมพันธ์ออก

#### 3.2 Scale Transformation (ต้องทำ)
```
✅ บางส่วน: Likert scale (1-5) มีอยู่แล้วในรูปแบบ numeric
⚠️ ต้องตรวจสอบ: มี columns ไหนบ้างที่เป็น Likert scale
```

**Likert scale columns:**
- factor_deep_cleansing
- factor_acne_friendly
- factor_sensitive_friendly
- factor_hypoallergenic
- factor_moisturizing
- factor_low_friction
- factor_nourishment
- factor_eye_friendly
- factor_oil_control

**สิ่งที่อาจต้องทำ:**
- พิจารณารวมเป็น composite score (เช่น "factor_average")
- หรือแบ่งกลุ่ม (สูง: 4-5, กลาง: 3, ต่ำ: 1-2)

#### 3.3 Encoding (ทำบางส่วนแล้ว)
```
✅ ทำแล้ว: gender, uses_cleansing_water_raw
❌ ยังไม่ได้ทำ: skin_type, acne_level, monthly_income, age
```

**ต้องเพิ่ม:**
- skin_type: Ordinal encoding หรือ One-hot
- acne_level: Ordinal encoding (ไม่มีสิว, เล็กน้อย, ปานกลาง, รุนแรง)
- monthly_income: Ordinal encoding (ตามช่วงรายได้)
- age: Ordinal encoding (ตามช่วงอายุ)

---

### 4. Data Visualization ⚠️ **ต้องทำเพิ่มทั้งหมด**

#### 4.1 Demographic Profile (ต้องทำ)
```python
# ต้องสร้าง visualization:
# - Bar chart: จำนวนผู้ตอบแยกตามเพศ
# - Bar chart: จำนวนผู้ตอบแยกตามช่วงอายุ
# - Bar chart: จำนวนผู้ตอบแยกตามอาชีพ
# - Pie chart: สัดส่วนประเภทผิว
# - Map หรือ Bar chart: จำนวนผู้ตอบแยกตามจังหวัด/ภูมิภาค
```

#### 4.2 Distribution Analysis (ต้องทำ)
```python
# ต้องสร้าง visualization:
# - Histogram: การกระจายของ Likert scale แต่ละ factor
# - Bar chart: สัดส่วนแบรนด์ที่ใช้ (brand_primary)
# - Bar chart: สัดส่วน cleansing type ที่ใช้
# - Stacked bar: ปัญหาผิวที่พบแยกตามประเภทผิว
```

#### 4.3 Correlation Heatmap (ต้องทำ)
```python
# ต้องสร้าง:
# - Heatmap แสดง correlation ระหว่าง factors ต่างๆ
# - Heatmap แสดง correlation ระหว่าง features กับ target
```

---

## สรุปสิ่งที่ต้องทำเพิ่ม (To-Do List)

### Priority 1: ต้องทำก่อน (สำคัญที่สุด)
1. **Data Cleaning**
   - [ ] Handle incomplete responses (จัดการ row ที่ข้อมูลไม่ครบ)
   - [ ] Standardization (จัดกลุ่ม occupation, province)
   - [ ] Logic check (ตรวจสอบความสมเหตุสมผล)

2. **Feature Engineering**
   - [ ] Encoding: skin_type, acne_level, monthly_income, age
   - [ ] แยก columns ที่เป็น multiple responses (skin_concerns, skincare_selection_method, ฯลฯ)

3. **Feature Selection**
   - [ ] Correlation analysis
   - [ ] Feature importance
   - [ ] เลือก features ที่มีผลต่อ target

### Priority 2: Visualization (ต้องทำสำหรับส่งงาน)
4. **Data Visualization**
   - [ ] Demographic Profile (5 plots)
   - [ ] Distribution Analysis (4-5 plots)
   - [ ] Correlation Heatmap (1-2 plots)

### Priority 3: เอกสารประกอบ
5. **Documentation**
   - [ ] ระบุเหตุผลที่เลือก target variable
   - [ ] อธิบาย logic ในการเลือก features
   - [ ] จัดทำ slide presentation

---

## โครงสร้างไฟล์ที่แนะนำ

```
final_pj_Kiyora/
├── data/
│   ├── data_prep.py          # มีอยู่แล้ว (ต้องแก้ไขเพิ่ม)
│   ├── dataset_cleansing.csv # มีอยู่แล้ว
│   └── dataset_prepared.csv  # มีอยู่แล้ว
├── visualization/
│   └── viz_analysis.py       # สร้างใหม่ (สำหรับ visualization)
├── docs/
│   └── target_definition.md  # สร้างใหม่ (อธิบาย target variable)
├── slides/
│   └── presentation.pptx     # สร้างใหม่ (สำหรับพรีเซนต์)
└── AIE323_Self_Learning_Analysis.md  # ไฟล์นี้
```

---

## หมายเหตุ

- **ห้ามแก้ไขโค้ดเดิม** ตามที่ผู้ใช้ขอ → จะสร้างไฟล์ใหม่หรือเพิ่ม comment แนะนำแทน
- **บันทึกใน Obsidian:** คัดลอกเนื้อหานี้ไปวางใน Obsidian vault ของคุณ

---

*สร้างเมื่อ: 2026-04-20*
*สำหรับรายวิชา: AIE323*
