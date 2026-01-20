import json
import re
import random

INPUT_FILE = "Zogoria_converted.json"
OUTPUT_FILE = "Zogoria_QA_clean.json"

FIXED_INSTRUCTION = "Soruyu yalnızca verilen eğitim bilgilerine dayanarak yanıtla."

# Basit soru kontrolü
def is_question(text: str) -> bool:
    if not text:
        return False
    return (
        text.strip().endswith("?")
        or text.lower().startswith(
            ("ne", "nedir", "nasıl", "neden", "hangi", "kaç", "kim", "nerede", "ne zaman")
        )
    )

# Soru kelimesi olmayan ama ? içerenleri de yakala
QUESTION_MARK_RE = re.compile(r"\?$")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

clean_data = []

for item in raw_data:
    instruction = item.get("instruction", "").strip()
    output = item.get("output", "").strip()

    # 1) Boş output → at
    if not output:
        continue

    # 2) Soru değilse → at
    if not is_question(instruction) and not QUESTION_MARK_RE.search(instruction):
        continue

    clean_item = {
        "instruction": FIXED_INSTRUCTION,
        "input": instruction,
        "output": output
    }

    clean_data.append(clean_item)

# 3) Rastgele "bilgi yok" örnekleri ekle (%10 oran)
UNKNOWN_ANSWERS = [
    "Bu bilgi eğitim verilerimde yer almıyor.",
    "Bu konuda eğitim verilerimde herhangi bir bilgi bulunmuyor.",
    "Bu soruya yanıt verecek bilgiye sahip değilim."
]

num_unknown = max(1, len(clean_data) // 10)

for _ in range(num_unknown):
    base = random.choice(clean_data)
    unknown_example = {
        "instruction": FIXED_INSTRUCTION,
        "input": base["input"].replace("nedir", "nerededir"),
        "output": random.choice(UNKNOWN_ANSWERS)
    }
    clean_data.append(unknown_example)

# 4) Kaydet
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, ensure_ascii=False, indent=2)

print(f"✅ Temizleme tamamlandı")
print(f"📦 Girdi kayıt sayısı : {len(raw_data)}")
print(f"🧼 Çıktı kayıt sayısı : {len(clean_data)}")
print(f"💾 Dosya kaydedildi  : {OUTPUT_FILE}")
