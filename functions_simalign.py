from simalign import SentenceAligner
import re


class WordNotInSentenceException(Exception):

    def __init__(self, word, sentence):
        self.word = word
        self.sentence = sentence


aligner = SentenceAligner(model="bert")


def get_position_in_sentence(word, sentence):
    splitting_chars = [
        " ",
        ",",
        "\\.",
        ":",
        ";",
        "!",
        "\\?",
        "'",
        "\\(",
        "\\)",
    ]  # TODO: clean out "-"; cannot split by them, but some of them left and right of the word can be removed
    words = [w for w in re.split("|".join(splitting_chars), sentence) if w != ""]
    if not word in words:
        raise WordNotInSentenceException(word=word, sentence=sentence)
    return words.index(word)


def add_pronouns(word, sentence):
    if re.match(f".*{word}[^a-zăâîșț-].*", sentence):
        return word
    return re.match(f".*({word}-?[a-zăâîșț]*-?[a-zăâîșț]*).*", sentence).group(1)


def do_words_correspond(word_en, sentence_en, word_ro, sentence_ro, alignments):
    position_ro = get_position_in_sentence(
        add_pronouns(word_ro, sentence_ro), sentence_ro
    )
    position_en = get_position_in_sentence(word_en, sentence_en)
    return (position_en, position_ro) in alignments


def is_match(word_en, sentence_en, word_ro, sentence_ro):
    alignments = aligner.get_word_aligns(sentence_en, sentence_ro)["mwmf"]
    errors = []

    if isinstance(word_en, list):
        for word in word_en:
            try:
                if do_words_correspond(
                    word, sentence_en, word_ro, sentence_ro, alignments
                ):
                    return True, errors
            except WordNotInSentenceException as e:
                print(f"Could not find word {e.word}!")
                errors.append(e)
                continue
        return False, errors

    elif isinstance(word_ro, list):
        for word in word_ro:
            try:
                if do_words_correspond(
                    word_en, sentence_en, word, sentence_ro, alignments
                ):
                    return True, errors
            except WordNotInSentenceException as e:
                print(f"Could not find word {e.word}!")
                errors.append(e)
                continue
        return False, errors

    else:
        try:
            return (
                do_words_correspond(
                    word_en, sentence_en, word_ro, sentence_ro, alignments
                ),
                errors,
            )
        except WordNotInSentenceException as e:
            print(f"Could not find word {e.word}!")
            return False, errors
