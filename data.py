import argparse
import logging
from pathlib import Path

from transformers import AutoTokenizer

from config.config import AugmentationConfig
from src.text_processing import TextProcessor
from src.utils import setup_logging, validate_file_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text augmentation and processing")
    parser.add_argument("--input-file", required=True, help="Input file path")
    parser.add_argument("--language", required=True, help="Language code")
    parser.add_argument(
        "--model-name", default="vinai/bartpho-syllable", help="Tokenizer model name"
    )
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument(
        "--test-size", type=int, default=20000, help="Number of samples for test set"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for processing"
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    setup_logging(args.log_file)

    try:
        # Validate input file
        input_path = validate_file_path(args.input_file)
        logging.info(f"Processing input file: {input_path}")

        # Initialize components
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        config = AugmentationConfig()
        config.BATCH_SIZE = args.batch_size

        processor = TextProcessor(tokenizer, config, args.language)

        # Set up output paths
        output_path = Path(f"{args.language}.csv")
        train_path = Path(f"{args.language}.train.csv")
        test_path = Path(f"{args.language}.test.csv")

        # Process file
        logging.info("Starting file processing...")
        skipped_lines = processor.process_file(input_path, output_path)
        logging.info(f"Processing complete. Output saved to {output_path}")
        logging.info(f"Skipped {skipped_lines} lines")

        # Split into train/test
        logging.info("Splitting into train and test sets...")
        processor.split_and_save(
            output_path, train_path, test_path, test_size=args.test_size
        )
        logging.info(f"Split complete. Train: {train_path}, Test: {test_path}")

    except Exception as e:
        logging.error(f"Error in main process: {e}")
        raise


if __name__ == "__main__":
    main()
