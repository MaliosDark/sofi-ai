eval:
	python eval_mteb.py MaliosDark/sofia-embedding-v1
	python eval_mteb.py ./SOFIA-v2

serve:
	uvicorn api:app --host 0.0.0.0 --port 8000

index:
	python build_index.py
	python search.py "best burgers"

train-lora:
	python train_lora_kd.py

infer:
	echo "machine learning is awesome" | python sofia_infer.py query

onnx:
	python export_onnx.py
