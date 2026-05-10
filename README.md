# Kiyora ML Dashboard

**AIE323 Self-Learning Assignment**

โปรเจกต์วิเคราะห์พฤติกรรมผู้ใช้ผลิตภัณฑ์คลีนซิ่ง ครอบคลุมตั้งแต่ Data Preparation, Supervised Learning (Random Forest / SVM), Unsupervised Learning (KMeans) ไปจนถึง REST API และ Dashboard ที่ deploy บน Vercel

**Live:** https://kiyora-dashboard-phi.vercel.app

---

## โครงสร้างโปรเจกต์

```
Kiyora/
├── app.py                          # FastAPI entrypoint (Vercel)
├── index.html                      # Dashboard UI (single-page)
├── requirements.txt
├── vercel.json
├── .env.example
│
├── backend/api/
│   ├── api.py                      # Route definitions
│   ├── analytics.py                # Data loading + model result serving
│   └── supabase_client.py
│
├── data/
│   └── data_prep_extended.py       # Data cleaning & feature engineering
│
├── model/
│   ├── sup.py                      # Supervised: Model A (binary) + Model B (cross-analysis)
│   ├── unsup.py                    # Unsupervised: KMeans segmentation
│   ├── sup_results.json            # ผลลัพธ์ Model A & B (auto-generated)
│   ├── kiyora_clustered.csv        # Cluster assignments (auto-generated)
│   └── kiyora_cluster_profile.csv  # Cluster feature means (auto-generated)
│
├── docs/
│   ├── supabase_schema.sql
│   └── target_definition.md
│
└── visualization/
    └── viz_analysis.py
```

---

## วิธีรัน

### 1. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 2. เตรียมข้อมูล

```bash
python data/data_prep_extended.py
```

สร้าง `dataset_extended_prepared.csv` ที่ root

### 3. Train models

```bash
# Supervised — สร้าง model/sup_results.json
python model/sup.py

# Unsupervised — สร้าง model/kiyora_clustered.csv + kiyora_cluster_profile.csv
python model/unsup.py
```

### 4. รัน API + Dashboard

```bash
# ตั้งค่า .env ก่อน (ดูหัวข้อ Supabase ด้านล่าง)
uvicorn app:app --reload
```

เปิด http://127.0.0.1:8000

---

## API Endpoints

| Method | Path | คำอธิบาย |
|--------|------|-----------|
| GET | `/` | Dashboard HTML |
| GET | `/api/health` | Health check |
| GET | `/api/overview` | ข้อมูลภาพรวม + model signals |
| GET | `/api/model/supervised` | ผลลัพธ์ Supervised Model (ต้องมี `sup_results.json`) |
| GET | `/api/model/unsupervised` | Cluster profiles ของ Kiyora users |
| GET | `/api/records` | ดึงข้อมูลจาก Supabase |
| POST | `/api/records` | บันทึกข้อมูลลง Supabase |

---

## Supabase (Optional)

1. สร้าง project ใน [supabase.com](https://supabase.com)
2. รัน `docs/supabase_schema.sql` ใน SQL Editor
3. สร้าง `.env` จาก `.env.example`:

```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_TABLE=Kiyora
```

ถ้าไม่มี Supabase API จะ fallback ไปอ่านจาก `dataset_extended_prepared.csv` อัตโนมัติ

---

## Deploy บน Vercel

```bash
vercel --prod
```

ตั้งค่า Environment Variables ใน Vercel Dashboard:

```
SUPABASE_URL
SUPABASE_KEY
SUPABASE_TABLE
```

---

## Models

### Supervised (Model A)
- **Task:** Binary classification — ทำนายว่าใครเป็นผู้ใช้ Kiyora
- **Best model:** Random Forest
- **Metrics:** Accuracy 90.9% · AUC 0.994 · CV-F1 0.917

### Supervised (Model B)
- **Task:** Cross-analysis Acne Level × Income Group → 12 segments
- **B1 (Acne):** SVM · **B2 (Income):** Random Forest

### Unsupervised
- **Task:** KMeans segmentation เฉพาะ Kiyora users (k=3)
- **Clusters:**
  - Cluster 0 — Everyday Gentle Users
  - Cluster 1 — Passive Legacy Users
  - Cluster 2 — Sensitive Acne Care Seekers
