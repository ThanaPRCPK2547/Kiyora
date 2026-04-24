# Kiyora

**AIE323 Self-Learning Assignment**

โปรเจกต์นี้เป็นการวิเคราะห์ข้อมูลผู้ใช้งานและพฤติกรรมการใช้ผลิตภัณฑ์คลีนซิ่ง (Cleansing) แบบครบวงจร ตั้งแต่ขั้นตอนการเตรียมข้อมูล (Data Preparation), การจัดการ Missing Values, การแปลงข้อมูล (Feature Encoding) ไปจนถึงการประยุกต์ใช้โมเดล Machine Learning (Random Forest) เพื่อทำนายและวิเคราะห์การเลือกใช้แบรนด์คลีนซิ่งหลัก (`brand_primary`) ของกลุ่มเป้าหมาย

## ภาพรวมของโปรเจกต์ (Project Overview)
- **กำหนดส่ง:** 27 เมษายน 2569
- **รายวิชา:** AIE323 
- **ตัวแปรเป้าหมาย (Target Variable):** `brand_primary` (แบรนด์คลีนซิ่งที่ผู้ใช้งานใช้บ่อยที่สุด)
- **เทคนิคที่ใช้หลัก:** `RandomForestClassifier` (ใช้ในการทำ Imputation ถมรอยโหว่ของข้อมูลและทำนายผล) พร้อมด้วยเทคนิค Feature Encoding เชิงลึก

## 📁 โครงสร้างโปรเจกต์ (Project Structure)
```text
final_pj_Kiyora/
├── README.md                         # ข้อมูลภาพรวมของโปรเจกต์
├── AIE323_Self_Learning_Analysis.md  # แผนการทำงานและรายละเอียด Todo-list เพื่อส่งอาจารย์
├── data/
│   ├── dataset_cleansing.csv         # ข้อมูลดิบตั้งต้นที่ได้จากการเก็บแบบสอบถาม
│   ├── dataset_prepared.csv          # ข้อมูลที่ผ่านการคลีน พร้อมใช้รันโมเดล (ML-Ready)
│   └── data_prep.py                  # สคริปต์หลักสำหรับ Data Cleaning และ Feature Engineering
├── model/
│   ├── sup.py                        # Supervised Learning Models
│   └── unsup.py                      # Unsupervised Learning Models
├── backend/                          # API และ Backend environment (หากต้องใช้)
├── frontend/                         # Web Interface (Dashboard หรือ UI)
└── deploy/                           # Deployment scripts & configs
```

## ฟีเจอร์หลักในการเตรียมข้อมูล (Data Pipeline Features)
- **Data Cleaning & Missing Value Handling:** จัดการตัวแปรและข้อมูลคำตอบที่ตกหล่น รวมถึงการแทนค่า `0` ให้กับกลุ่มคนที่ไม่ได้ใช้งานคลีนซิ่งเพื่อให้การวิเคราะห์แม่นยำขึ้น
- **In-place Feature Engineering:** สคริปต์สามารถแปลงข้อมูลคำตอบทางสถิติของฟีเจอร์หลัก (`age`, `monthly_income`, `acne_level`, `skin_type`) ให้กลายเป็นตัวเลข (Ordinal & Label Encoding) โดยที่**ยังคงรักษาชื่อคอลัมน์และลำดับของโครงสร้างตารางเอาไว้ 100%**
- **Automated Missing Data Prediction:** สามารถพยากรณ์และทายผลของคนที่ตอบค่าเป้าหมาย (`brand_primary`) ไว้ว่าเป็นค่าว่างได้สำเร็จ 

## วิธีการใช้งาน (How to Run)

1. รันไฟล์เพื่อเตรียมข้อมูล คลีน และเข้ารหัสข้อมูล (Data Preparation):
```bash
cd data
python3 data_prep.py
```
2. ไฟล์ `dataset_prepared.csv` ซึ่งสมบูรณ์และไม่มีปัญหาการประมวลผล (ไม่มีค่า Null/Error ใน features หลัก) จะถูกเซฟออกมาโดยอัตโนมัติ 

## แผนงานต่อไปเพื่อส่งอาจารย์ (Next Steps)
- ทำ **Data Visualization** (พล็อตกราฟ Demographic Profiles, Distribution Analysis, รัน Correlation Heatmap หาตัวแปรเสริม)
- จัดพิมพ์ **Documentation / Presentation** ขยายความถึงที่มาการเลือกตัวแปรเป้าหมาย
