import csv
import tempfile
import unittest
from pathlib import Path

from transformers import AutoTokenizer

from config.config import AugmentationConfig
from src.augmentation import TextAugmenter
from src.text_processing import TextProcessor


class TestTextAugmenter(unittest.TestCase):
    """Test cases for text augmentation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig()
        self.augmenter = TextAugmenter(self.config)
        self.test_text = "Hello World! This is a test."

    def test_delete_characters(self):
        """Test character deletion."""
        result = self.augmenter.delete_characters(self.test_text)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) <= len(self.test_text))

    def test_insert_characters(self):
        """Test character insertion."""
        result = self.augmenter.insert_characters(self.test_text)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) >= len(self.test_text))

    def test_replace_characters(self):
        """Test character replacement."""
        result = self.augmenter.replace_characters(self.test_text)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), len(self.test_text))

    def test_swap_characters_case(self):
        """Test character case swapping."""
        result = self.augmenter.swap_characters_case(self.test_text)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), len(self.test_text))


class TestTextProcessor(unittest.TestCase):
    """Test cases for text processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig()
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
        self.processor = TextProcessor(self.tokenizer, self.config, "en")

    def test_process_file(self):
        """Test file processing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_input:
            # Write test data
            temp_input.write("Test line 1\nTest line 2\nTest line 3\n")
            temp_input_path = Path(temp_input.name)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_output:
            temp_output_path = Path(temp_output.name)

        try:
            self.processor.process_file(temp_input_path, temp_output_path)

            # Verify output
            with open(temp_output_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(header, ["input", "output"])
                rows = list(reader)
                self.assertTrue(len(rows) > 0)

        finally:
            # Cleanup
            temp_input_path.unlink()
            temp_output_path.unlink()

    def test_split_and_save(self):
        """Test splitting data into train and test sets."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_input:
            writer = csv.writer(temp_input)
            writer.writerow(["input", "output"])
            for i in range(100):
                writer.writerow([f"input_{i}", f"output_{i}"])
            temp_input_path = Path(temp_input.name)

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False
        ) as temp_train, tempfile.NamedTemporaryFile(
            mode="w", delete=False
        ) as temp_test:
            temp_train_path = Path(temp_train.name)
            temp_test_path = Path(temp_test.name)

        try:
            self.processor.split_and_save(
                temp_input_path, temp_train_path, temp_test_path, test_size=20
            )

            # Verify split
            with open(temp_train_path, "r") as f:
                train_rows = list(csv.reader(f))
            with open(temp_test_path, "r") as f:
                test_rows = list(csv.reader(f))

            self.assertEqual(len(test_rows) - 1, 20)  # -1 for header
            self.assertEqual(len(train_rows) - 1, 80)  # -1 for header

        finally:
            # Cleanup
            temp_input_path.unlink()
            temp_train_path.unlink()
            temp_test_path.unlink()


if __name__ == "__main__":
    unittest.main()
