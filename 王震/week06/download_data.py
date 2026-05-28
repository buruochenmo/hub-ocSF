
import json


data = [
    {"sentence": "今天天气真好", "label": 0},
    {"sentence": "股市大跌，经济不好", "label": 1},
    # ...
]

with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
