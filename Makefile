lint:
	pre-commit run --all-files


test:
	python -m unittest -v -f test_resnet_rs/test*.py
