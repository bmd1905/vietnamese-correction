import csv
import logging
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm
from transformers import PreTrainedTokenizer

from config.config import AugmentationConfig

from .augmentation import TextAugmenter
from .utils import is_text_too_long


class TextProcessor:
    """Class for processing and augmenting text data.

    This class is responsible for handling the processing of text data,
    including reading from input files, augmenting the text using various
    techniques, and writing the results to output files. It utilizes the
    TextAugmenter class to perform augmentations based on a specified
    configuration.

    Attributes:
        tokenizer (PreTrainedTokenizer): A tokenizer for processing text.
        config (AugmentationConfig): Configuration object containing parameters
            for text augmentation.
        language (str): The language of the text being processed.
        augmenter (TextAugmenter): An instance of the TextAugmenter class
            for augmenting text data.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, config: AugmentationConfig, language: str
    ):
        """Initialize the TextProcessor with the given tokenizer, configuration, and language.

        Args:
            tokenizer (PreTrainedTokenizer): A tokenizer for processing text.
            config (AugmentationConfig): Configuration object for text augmentation.
            language (str): The language of the text being processed.
        """
        self.tokenizer = tokenizer
        self.config = config
        self.language = language
        self.augmenter = TextAugmenter(config)

    def process_batch(self, batch: List[str], writer: csv.writer) -> int:
        """Process a batch of text entries and write results to the provided CSV writer.

        This method iterates over a batch of text lines, checks if each line
        is valid (not empty and within the maximum length), and writes the
        original and augmented lines to the CSV writer.

        Args:
            batch (List[str]): A list of text lines to process.
            writer (csv.writer): A CSV writer object to write results.

        Returns:
            int: The number of lines that were skipped during processing.
        """
        skipped_lines = 0

        for line in batch:
            line = line.strip().replace('"', "")
            if not line or is_text_too_long(
                line, self.tokenizer, self.config.MAX_LENGTH
            ):
                skipped_lines += 1
                continue

            try:
                # Write the original line
                writer.writerow([line, line])

                # Augment the line based on the configuration
                for _ in range(self.config.AUGMENTATIONS_PER_SAMPLE):
                    augmented_line = self.augmenter.augment_text(line)
                    writer.writerow([augmented_line, line])
            except Exception as e:
                logging.error(f"Error processing line: {e}")

        return skipped_lines

    def process_file(
        self, input_file: Path, output_file: Path, batch_size: Optional[int] = None
    ) -> None:
        """Process an input file and write the results to an output file.

        This method reads the input file line by line, processes the lines
        in batches, and writes the results to the output file. It handles
        any exceptions that may occur during file operations.

        Args:
            input_file (Path): The path to the input file containing text data.
            output_file (Path): The path to the output file where results will be written.
            batch_size (Optional[int]): The number of lines to process in each batch.
                If None, the default batch size from the configuration will be used.
        """
        batch_size = batch_size or self.config.BATCH_SIZE
        skipped_lines = 0
        try:
            with open(input_file, "r", encoding="utf-8") as infile, open(
                output_file, "w", newline="", encoding="utf-8"
            ) as outfile:
                writer = csv.writer(outfile)
                writer.writerow(["input", "output"])

                current_batch = []
                for line in tqdm(infile):
                    current_batch.append(line)
                    if len(current_batch) >= batch_size:
                        skipped_lines += self.process_batch(current_batch, writer)
                        current_batch = []

                if current_batch:
                    skipped_lines += self.process_batch(current_batch, writer)

        except Exception as e:
            logging.error(f"Error processing file: {e}")
            raise

        return skipped_lines

    def split_and_save(
        self, input_file: Path, train_file: Path, test_file: Path, test_size: int
    ) -> None:
        """Split the processed data into training and testing sets.

        This method reads the processed data from the input file, splits it
        into training and testing sets based on the specified test size, and
        writes the results to the respective training and testing files.

        Args:
            input_file (Path): The path to the input file containing processed data.
            train_file (Path): The path to the output file for training data.
            test_file (Path): The path to the output file for testing data.
            test_size (int): The desired size of the test set.
        """
        with open(input_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)

        # Adjust test_size to account for augmented data
        samples_per_original = self.config.AUGMENTATIONS_PER_SAMPLE + 1
        original_test_size = test_size // samples_per_original
        test_data = data[: original_test_size * samples_per_original]
        train_data = data[original_test_size * samples_per_original :]

        with open(train_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(train_data)

        with open(test_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(test_data)
