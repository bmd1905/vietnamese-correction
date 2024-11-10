from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AugmentationConfig:
    """Configuration for text augmentation parameters."""

    AUGMENTATION_PROBABILITY: float = 0.01
    CHAR_DELETE_PERCENTAGE: float = 0.01
    LOWER_CASE_WORDS_PROBABILITY: float = 0.1
    COMBINE_SENTENCES_PROBABILITY: float = 1
    DELETE_WORD_PROBABILITY: float = 0.005
    REPLACE_ACCENT_CHARS_RATIO: float = 0.02
    REMOVE_RANDOM_ACCENT_RATIO: float = 0.1
    BATCH_SIZE: int = 1_000
    MAX_LENGTH: int = 256
    AUGMENTATIONS_PER_SAMPLE: int = 5


@dataclass
class CharacterMaps:
    """Mapping for character replacements and regex patterns."""

    SAME_CHARS: Dict[str, List[str]] = None
    CHARS_REGEX: str = r"[aàảãáạăằẳẵắặâầẩẫấậAÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬoòỏõóọôồổỗốộơờởỡớợOÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỢeèẻẽéẹêềểễếệEÈẺẼÉẸÊỀỂỄẾỆuùủũúụưừửữứựUÙỦŨÚỤƯỪỬỮỨỰiìỉĩíịIÌỈĨÍỊyỳỷỹýỵYỲỶỸÝỴnNvVmMCG]"

    def __post_init__(self):
        self.SAME_CHARS = {
            "a": [
                "á",
                "à",
                "ả",
                "ã",
                "ạ",
                "ấ",
                "ầ",
                "ẩ",
                "ẫ",
                "ậ",
                "ắ",
                "ằ",
                "ẳ",
                "ẵ",
                "ặ",
            ],
            "A": [
                "Á",
                "À",
                "Ả",
                "Ã",
                "Ạ",
                "Ấ",
                "Ầ",
                "Ẩ",
                "Ẫ",
                "Ậ",
                "Ắ",
                "Ằ",
                "Ẳ",
                "Ẵ",
                "Ặ",
            ],
            "O": [
                "Ó",
                "Ò",
                "Ỏ",
                "Õ",
                "Ọ",
                "Ô",
                "Ố",
                "Ồ",
                "Ổ",
                "Ỗ",
                "Ộ",
                "Ơ",
                "Ớ",
                "Ờ",
                "Ở",
                "Ỡ",
                "Ợ",
                "Q",
            ],
            "o": [
                "ó",
                "ò",
                "ỏ",
                "õ",
                "ọ",
                "ô",
                "ố",
                "ồ",
                "ổ",
                "ỗ",
                "ộ",
                "ơ",
                "ớ",
                "ờ",
                "ở",
                "ỡ",
                "ợ",
                "q",
            ],
            "e": ["é", "è", "ẻ", "ẽ", "ẹ", "ế", "ề", "ể", "ễ", "ệ", "ê"],
            "E": ["É", "È", "Ẻ", "Ẽ", "Ẹ", "Ế", "Ề", "Ể", "Ễ", "Ệ", "Ê"],
            "u": ["ú", "ù", "ủ", "ũ", "ụ", "ứ", "ừ", "ử", "ữ", "ự", "ư"],
            "U": ["Ú", "Ù", "Ủ", "Ũ", "Ụ", "Ứ", "Ừ", "Ử", "Ữ", "Ự", "Ư"],
            "i": ["í", "ì", "ỉ", "ĩ", "ị"],
            "I": ["Í", "Ì", "Ỉ", "Ĩ", "Ị"],
            "y": ["ý", "ỳ", "ỷ", "ỹ", "ỵ", "v"],
            "Y": ["Ý", "Ỳ", "Ỷ", "Ỹ", "Ỵ", "V"],
            "n": ["m"],
            "N": ["N"],
            "v": ["y"],
            "V": ["Y"],
            "m": ["n"],
            "M": ["N"],
            "C": ["G"],
            "G": ["C"],
        }
