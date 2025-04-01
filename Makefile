install:
	poetry env use 3.11.0
	poetry lock
	poetry install
	@if [ -n "$$CUDA_HOME" ]; then \
		echo "\033[0;32mCUDA_HOME is set to $$CUDA_HOME, installing flash-attn\033[0m"; \
	    poetry run pip install flash_attn --no-build-isolation; \
	else \
		echo "\033[0;31mCUDA_HOME is not set, skipping flash-attn installation\033[0m"; \
	fi