.PHONY: data test


data:
	poetry run python3 data.py --input-file ./data/data.vi.txt --language vi --test-size 100000

test:
	poetry run python -m unittest tests/test_augmentation.py