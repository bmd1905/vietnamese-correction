import logging
from pathlib import Path
from typing import Optional, Union

from transformers import PreTrainedTokenizer


def setup_logging(log_file: Optional[str] = None) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
        ],
    )


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """Validate if file path exists and is accessible."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def is_text_too_long(
    text: str, tokenizer: PreTrainedTokenizer, max_length: int
) -> bool:
    """Check if text exceeds maximum length after tokenization."""
    try:
        return len(tokenizer.encode(text)) > max_length
    except Exception as e:
        logging.error(f"Error tokenizing text: {e}")
        return True
