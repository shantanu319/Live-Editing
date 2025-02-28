import os
import openai
import nltk
from nltk.corpus import words
import re

class SpellChecker:
    def __init__(self):
        nltk.download("words", quiet=True)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = openai.OpenAI(api_key=api_key)
        self.word_set = set(words.words()) 
    

    def correct_spelling(self, word):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a spell checker. Respond only with the corrected word."},
                    {"role": "user", "content": f"Correct the spelling of: {word}"}
                ],
                max_tokens=50,
                temperature=0.0  # Use deterministic output
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error correcting word: {e}")
            return word

    def check_text(self, text):
        """Check and correct spelling in the given text."""
        # Split text into words, preserving spaces and punctuation
        words_with_positions = []
        for match in re.finditer(r'\S+', text):
            words_with_positions.append(
                (match.group(), match.start(), match.end()))

        corrected_text = list(text)
        corrections_made = []

        for word, start, end in words_with_positions:

            clean_word = re.sub(r'[^\w]', '', word.lower())

            # Skip empty strings
            if len(clean_word) <= 1:
                continue

            # Check if word is misspelled
            if clean_word not in self.word_set:
                corrected_word = self.correct_spelling(word)
                if corrected_word != word:
                    # Replace the word
                    corrected_text[start:end] = corrected_word
                    corrections_made.append((word, corrected_word))
        return ''.join(corrected_text), corrections_made


def main():
    checker = SpellChecker()
    print("Welcome to the Spell Checker! (Type 'exit' to quit)")

    while True:
        text = input("\nEnter text to check: ")
        if text.lower() == 'exit':
            break

        corrected_text, corrections = checker.check_text(text)

        if corrections:
            print("\nCorrected text:", corrected_text)
            print("\nCorrections made:")
            for original, corrected in corrections:
                print(f"  '{original}' → '{corrected}'")
        else:
            print("No spelling errors found!")


if __name__ == "__main__":
    main()
