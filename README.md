# Kiyora Data Analysis Pipeline

**AIE323 Self-Learning Assignment**

โปรเจกต์นี้เป็นการวิเคราะห์ข้อมูลผู้ใช้งานและพฤติกรรมการใช้ผลิตภัณฑ์คลีนซิ่ง (Cleansing) แบบครบวงจร ตั้งแต่ขั้นตอนการเตรียมข้อมูล (Data Preparation), การจัดการ Missing Values, การแปลงข้อมูล (Feature Encoding) ไปจนถึงการประยุกต์ใช้โมเดล Machine Learning (Random Forest) เพื่อทำนายและวิเคราะห์การเลือกใช้แบรนด์คลีนซิ่งหลัก (`brand_primary`) ของกลุ่มเป้าหมาย

## ภาพรวมของโปรเจกต์ (Project Overview)
- **กำหนดส่ง:** 27 เมษายน 2569
- **รายวิชา:** AIE323 
- **ตัวแปรเป้าหมาย (Target Variable):** `brand_primary` (แบรนด์คลีนซิ่งที่ผู้ใช้งานใช้บ่อยที่สุด)
- **เทคนิคที่ใช้หลัก:** `RandomForestClassifier` (ใช้ในการทำ Imputation ถมรอยโหว่ของข้อมูลและทำนายผล) พร้อมด้วยเทคนิค Feature Encoding เชิงลึก และสถิติวิเคราะห์ (Correlation)

## โครงสร้างโปรเจกต์ (Project Structure)
```text
final_pj_Kiyora/
├── README.md                         # ข้อมูลภาพรวมของโปรเจกต์
├── AIE323_Self_Learning_Analysis.md  # แผนการทำงานและรายละเอียด Todo-list เพื่อส่งอาจารย์
├── docs/
│   └── target_definition.md          # เอกสารอธิบายเหตุผลการเลือก Target Variable
├── data/
│   ├── dataset_cleansing.csv         # ข้อมูลดิบตั้งต้นที่ได้จากการเก็บแบบสอบถาม
│   ├── dataset_prepared.csv          # ข้อมูลที่ผ่านการคลีนเบื้องต้น
│   ├── dataset_extended_prepared.csv # ข้อมูลที่ผ่านการคลีนและทำ Feature Engineering ฉบับสมบูรณ์
│   ├── data_prep.py                  # สคริปต์หลักสำหรับ Data Cleaning
│   └── data_prep_extended.py         # สคริปต์เตรียมข้อมูลฉบับสมบูรณ์พร้อม Feature Selection
├── visualization/
│   ├── viz_analysis.py               # สคริปต์สำหรับพล็อตกราฟและ Data Visualization
│   └── plots/                        # โฟลเดอร์เก็บรูปภาพกราฟผลลัพธ์
├── model/
│   ├── sup.py                        # Supervised Learning Models
│   └── unsup.py                      # Unsupervised Learning Models
├── backend/                          # API และ Backend environment (หากต้องใช้)
├── frontend/                         # Web Interface (Dashboard หรือ UI)
└── deploy/                           # Deployment scripts & configs
```

## ฟีเจอร์หลักในการเตรียมข้อมูลและการวิเคราะห์ (Data Pipeline Features)
- **Data Cleaning & Missing Value Handling:** จัดการตัวแปรและข้อมูลคำตอบที่ตกหล่น รวมถึงการแทนค่า `0` ให้กับกลุ่มคนที่ไม่ได้ใช้งานคลีนซิ่งเพื่อให้การวิเคราะห์แม่นยำขึ้น
- **In-place Feature Engineering:** สคริปต์สามารถแปลงข้อมูลคำตอบทางสถิติของฟีเจอร์หลักให้กลายเป็นตัวเลข (Ordinal & Label Encoding) และแยกคำตอบแบบเลือกได้หลายข้อ (Multiple Responses) ให้อยู่ในรูปแบบ One-Hot Encoding
- **Automated Missing Data Prediction:** สามารถพยากรณ์และทายผลของคนที่ตอบค่าเป้าหมาย (`brand_primary`) ไว้ว่าเป็นค่าว่างได้สำเร็จ โดยใช้ Random Forest
- **Data Visualization & Insights:** สร้างกราฟวิเคราะห์ประชากรศาสตร์ (Demographics), สัดส่วนแบรนด์หลัก, และระดับความสำคัญของคุณสมบัติคลีนซิ่ง พร้อมทั้งสร้างแผนภาพความสัมพันธ์ (Correlation Heatmap) ระหว่างฟีเจอร์ต่างๆ และตัวแปรเป้าหมาย

## วิธีการใช้งาน (How to Run)

1. **ติดตั้งไลบรารีที่จำเป็น:**
```bash
pip install -r requiment.txt
```

2. **รันไฟล์เพื่อเตรียมข้อมูล คลีน และเข้ารหัสข้อมูล (Data Preparation):**
```bash
cd data
python3 data_prep_extended.py
```
*(ไฟล์ `dataset_extended_prepared.csv` จะถูกสร้างขึ้นมาใน Root Directory)*

3. **รันไฟล์เพื่อสร้าง Data Visualization:**
```bash
cd ../visualization
python3 viz_analysis.py
```
*(รูปภาพกราฟทั้งหมดจะถูกบันทึกในโฟลเดอร์ `visualization/plots/`)*
