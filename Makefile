install:
	pip install -r requirements.txt

prepare-data:
	python -m citrusnet.data.prepare_data

test:
	pytest -q

check-models:
	python scripts/check_model_size.py

train:
	python -m citrusnet.training.train --config configs/train.yaml

pretrain:
	python -m citrusnet.ssl_bt.pretrain --config configs/ssl_bt.yaml