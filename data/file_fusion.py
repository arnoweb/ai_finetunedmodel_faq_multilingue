import json

# Fichiers d'entrée
input_files = ["faq_source_fr.jsonl", "faq_source_en.jsonl"]
output_file = "faq_train.jsonl"

with open(output_file, "w", encoding="utf-8") as fout:
    for file in input_files:
        with open(file, "r", encoding="utf-8") as fin:
            for line in fin:
                item = json.loads(line)
                # On enlève la clé "language" si elle existe (facultatif)
                item.pop("language", None)
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ File merged: faq_train.jsonl")
