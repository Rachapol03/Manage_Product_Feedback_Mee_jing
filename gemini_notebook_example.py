# ตัวอย่าง Notebook Cells สำหรับใช้ Gemini API

# ========== CELL 1: ติดตั้ง Dependencies ==========
# !pip install python-dotenv google-generativeai pandas

# ========== CELL 2: โหลด Libraries ==========
import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# ========== CELL 3: ตั้งค่า Environment Variables ==========
# โหลด .env file
load_dotenv()

# ดึง API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

# ตั้งค่า Gemini
genai.configure(api_key=GEMINI_API_KEY)

print(f"✅ API Key ถูกโหลดสำเร็จ")
print(f"📌 Model: {GEMINI_MODEL}")

# ========== CELL 4: สร้าง Function ประมวลผล Single Row ==========
def analyze_comment_with_gemini(comment: str) -> str:
    """วิเคราะห์ความเห็นเดียวโดยใช้ Gemini"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    prompt = f"""
    โปรดวิเคราะห์ความเห็นต่อไปนี้:
    \"{comment}\"
    
    ให้ผลลัพธ์:
    1. ความรู้สึก (Sentiment): บวก/กลาง/ลบ
    2. หัวข้อหลัก (Topic): 
    3. ความสำคัญ (Priority): สูง/ปานกลาง/ต่ำ
    """
    
    response = model.generate_content(prompt)
    return response.text

# ========== CELL 5: โหลดข้อมูล CSV ==========
# ปรับ column name ตามไฟล์ของคุณ
df = pd.read_csv("comments_youtube.csv")
print(f"📊 โหลดข้อมูล: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(df.head())

# ========== CELL 6: ประมวลผล Row ทีละตัว ==========
# ตัวอย่างการประมวลผล 3 rows แรก
responses = []

for idx, row in df.head(3).iterrows():
    comment = row["comment"]  # ปรับ column name ตามของคุณ
    print(f"\n🔄 วิเคราะห์ row {idx + 1}...")
    print(f"💬 ความเห็น: {comment[:100]}...")
    
    response = analyze_comment_with_gemini(comment)
    responses.append(response)
    print(f"📝 ผลลัพธ์:\n{response}")

# ========== CELL 7: เพิ่มผลลัพธ์เป็น Column ใหม่ ==========
# ประมวลผลทั้งหมด
print("⏳ กำลังประมวลผลทั้งหมด...")
df["gemini_analysis"] = df["comment"].apply(analyze_comment_with_gemini)

print("✅ เสร็จแล้ว!")
print(df[["comment", "gemini_analysis"]].head())

# ========== CELL 8: บันทึกผลลัพธ์ ==========
output_file = "comments_with_analysis.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"💾 บันทึกข้อมูลเสร็จในไฟล์: {output_file}")

# ========== CELL 9: ตัวอย่างการใช้ Streaming Response (เร็วขึ้น) ==========
def analyze_comment_streaming(comment: str) -> str:
    """วิเคราะห์โดยใช้ Streaming"""
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    prompt = f"วิเคราะห์เหตุผลหลักของความเห็นนี้: {comment}"
    
    response = ""
    for chunk in model.generate_content(prompt, stream=True):
        response += chunk.text
    
    return response

# ========== Optional: CELL 10: บันทึก API Key เข้า .env (ครั้งแรกเท่านั้น) ==========
# สำหรับการใช้งานครั้งแรก: แก้ไขไฟล์ .env และเพิ่ม:
# GEMINI_API_KEY=your_actual_api_key_here
# GEMINI_MODEL=gemini-1.5-pro  # หรือ gemini-pro
