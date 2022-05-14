lint:
	pre-commit run --all-files


test:
	pytest -x test_resnet_rs/test*.py  # Run all tests except check_output_consistency.py
