import json
cn = {int(k): v for k, v in json.load(open("data/processed/category_names.json")).items()}
m = json.load(open("data/processed/metadata.json"))
print("TOTAL CATS:", len(cn))
print("train(%d) val(%d) test(%d)" % (
    len(m["train_categories"]), len(m["val_categories"]), len(m["test_categories"])))
print("TEST:", [cn[c] for c in m["test_categories"]])
print("TRAIN sample:", [cn[c] for c in m["train_categories"][:10]])
