import json
from pathlib import Path

DATA_DIR = Path("data")
FORMATTED_DIR = Path(f"{DATA_DIR}/formatted")
FORMATTED_DIR.mkdir(exist_ok=True)
INSTRUCTION = "Beantworte die folgende Frage."

for file in DATA_DIR.glob("*.json"):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "qa_pairs" in data:
        qa_pairs = data["qa_pairs"]
    elif isinstance(data, list):
        qa_pairs = data
    else:
        print(f"⚠️  Skipping {file.name} - unknown format.")
        continue

    converted = []
    for pair in qa_pairs:
        question = pair.get("question", "").strip()
        answer = pair.get("answer", "").strip()

        converted.append({
            "instruction": INSTRUCTION,
            "input": question,
            "output": answer
        })

    outfile = FORMATTED_DIR / file.name
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
