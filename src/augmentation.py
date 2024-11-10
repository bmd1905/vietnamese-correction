import logging
import random
import re
from string import ascii_letters

import unidecode

from config.config import AugmentationConfig, CharacterMaps


class TextAugmenter:
    """Class for text augmentation operations.

    This class provides various methods to augment text data by applying
    different transformations such as character case swapping, character
    deletion, insertion, replacement, and accent modification. The
    transformations are probabilistic and configurable through the
    AugmentationConfig class.

    Attributes:
        config (AugmentationConfig): Configuration object containing
            parameters for augmentation techniques.
        char_maps (CharacterMaps): Object containing character mapping
            information for accent modifications.
        clean_punctuation (Pattern): Compiled regex pattern for cleaning
            punctuation from text.
    """

    def __init__(self, config: AugmentationConfig):
        """Initialize the TextAugmenter with the given configuration.

        Args:
            config (AugmentationConfig): Configuration object for text
                augmentation parameters.
        """
        self.config = config
        self.char_maps = CharacterMaps()
        self.clean_punctuation = re.compile(r"(?<!\d)[.,;:'?!](?!\d)")

    def augment_text(self, text: str) -> str:
        """Apply all augmentation techniques to the provided text.

        This method applies a series of transformations to the input text
        in a specific order. If any error occurs during the augmentation
        process, it logs the error and returns the original text.

        Args:
            text (str): The input text to be augmented.

        Returns:
            str: The augmented text after applying all transformations.
        """
        try:
            text = self.swap_characters_case(text)
            text = self.delete_word(text)
            text = self.delete_characters(text)
            text = self.insert_characters(text)
            text = self.replace_characters(text)
            text = self.lower_case_words(text)
            text = self.remove_punctuation(text)
            text = self.remove_random_accent(text)
            text = self.replace_accent_chars(text)
            return text
        except Exception as e:
            logging.error(f"Error during text augmentation: {e}")
            return text

    def delete_characters(self, text: str) -> str:
        """Delete random characters from the input text.

        This method removes characters from the text based on a
        configurable probability. Digits are not removed.

        Args:
            text (str): The input text from which characters will be deleted.

        Returns:
            str: The modified text with random characters deleted.
        """
        return "".join(
            c
            for c in text
            if random.random() > self.config.CHAR_DELETE_PERCENTAGE or c.isdigit()
        )

    def insert_characters(self, text: str) -> str:
        """Insert random characters into the input text.

        This method inserts random ASCII letters into the text based on
        a configurable probability. Digits are not affected.

        Args:
            text (str): The input text into which characters will be inserted.

        Returns:
            str: The modified text with random characters inserted.
        """
        modified_line = []
        for char in text:
            if (
                random.random() <= self.config.AUGMENTATION_PROBABILITY
                and not char.isdigit()
            ):
                modified_line.append(random.choice(ascii_letters))
            modified_line.append(char)
        return "".join(modified_line)

    def replace_characters(self, text: str) -> str:
        """Replace characters in the input text with random ASCII letters.

        This method replaces characters in the text based on a
        configurable probability. Digits are not replaced.

        Args:
            text (str): The input text in which characters will be replaced.

        Returns:
            str: The modified text with characters replaced by random ASCII letters.
        """
        return "".join(
            random.choice(ascii_letters)
            if random.random() <= self.config.AUGMENTATION_PROBABILITY
            and not c.isdigit()
            else c
            for c in text
        )

    def swap_characters_case(self, text: str) -> str:
        """Swap the case of characters in the input text randomly.

        This method changes uppercase characters to lowercase and vice versa
        based on a configurable probability.

        Args:
            text (str): The input text in which character cases will be swapped.

        Returns:
            str: The modified text with character cases swapped.
        """
        return "".join(
            c.swapcase()
            if random.random() <= self.config.AUGMENTATION_PROBABILITY
            else c
            for c in text
        )

    def lower_case_words(self, text: str) -> str:
        """Convert the first letter of words to lowercase randomly.

        This method changes the first letter of words to lowercase based
        on a configurable probability.

        Args:
            text (str): The input text in which words will be modified.

        Returns:
            str: The modified text with words converted to lowercase.
        """
        return " ".join(
            word.lower()
            if word[0].isupper()
            and random.random() <= self.config.LOWER_CASE_WORDS_PROBABILITY
            else word
            for word in text.split()
        )

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from the input text.

        This method uses a compiled regex pattern to remove punctuation
        characters from the text.

        Args:
            text (str): The input text from which punctuation will be removed.

        Returns:
            str: The modified text with punctuation removed.
        """
        return self.clean_punctuation.sub("", text)

    def delete_word(self, text: str) -> str:
        """Delete a random word from the input text.

        This method removes a word from the text based on a configurable
        probability, ensuring that at least two words remain.

        Args:
            text (str): The input text from which a word will be deleted.

        Returns:
            str: The modified text with a random word deleted, or the
            original text if no deletion occurs.
        """
        words = text.split()
        if len(words) >= 3 and random.random() < self.config.DELETE_WORD_PROBABILITY:
            words.pop(random.randint(0, len(words) - 1))
            return " ".join(words)
        return text

    def replace_accent_chars(self, text: str) -> str:
        """Replace accented characters in the input text.

        This method replaces a randomly selected accented character in a
        word with a different character from the same group based on a
        configurable probability.

        Args:
            text (str): The input text in which accented characters will be replaced.

        Returns:
            str: The modified text with accented characters replaced.
        """
        words = text.split()
        if random.random() < self.config.REPLACE_ACCENT_CHARS_RATIO and words:
            idx = random.randint(0, len(words) - 1)
            words[idx] = self._change_accent(words[idx])
        return " ".join(words)

    def remove_random_accent(self, text: str) -> str:
        """Remove accents from random words in the input text.

        This method removes accents from a randomly selected word based
        on a configurable probability.

        Args:
            text (str): The input text from which accents will be removed.

        Returns:
            str: The modified text with accents removed from random words.
        """
        words = text.split()
        if random.random() < self.config.REMOVE_RANDOM_ACCENT_RATIO and words:
            idx = random.randint(0, len(words) - 1)
            words[idx] = unidecode.unidecode(words[idx])
        return " ".join(words)

    def _change_accent(self, text: str) -> str:
        """Change accents in the given text.

        This helper method finds accented characters in the text and
        replaces one of them with a different character from the same
        group based on a random selection.

        Args:
            text (str): The input text in which accents will be changed.

        Returns:
            str: The modified text with accents changed, or the original
            text if no changes are made.
        """
        match_chars = re.findall(self.char_maps.CHARS_REGEX, text)
        if not match_chars:
            return text

        replace_char = random.choice(match_chars)
        base_char = unidecode.unidecode(replace_char)
        if base_char in self.char_maps.SAME_CHARS:
            insert_char = random.choice(self.char_maps.SAME_CHARS[base_char])
            return text.replace(replace_char, insert_char, 1)
        return text
