import ast
for f in ["src/eval_seen.py", "src/evaluate.py", "src/train.py", "src/export_tflite.py", "src/data_loader.py"]:
    try:
        ast.parse(open(f).read())
        print("OK  ", f)
    except SyntaxError as e:
        print("FAIL", f, "->", e)
