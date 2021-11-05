lint:
	pre-commit run --all-files


test:
	@for f in $(shell ls test_resnet_rs/test*.py); do \
  		echo $${f};\
		python $${f};\
		done
