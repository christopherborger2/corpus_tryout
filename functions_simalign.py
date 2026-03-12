from simalign import SentenceAligner
import re
import pandas as pd
import pickle

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
        "’",
    ]  # TODO: clean out "-"; cannot split by them, but some of them left and right of the word can be removed
    words = [w for w in re.split("|".join(splitting_chars), sentence) if w != ""]
    if not word in words:
        raise RuntimeError({"word": word, "sentence": sentence})
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
            except RuntimeError as e:
                print(f"Could not find word {e.args[0]["word"]}!")
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
            except RuntimeError as e:
                print(f"Could not find word {e.args[0]["word"]}!")
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
        except RuntimeError as e:
            print(f"Could not find word {e.args[0]["word"]}!")
            return False, errors


def match_adv_part_and_gerunzius(df, language):

    if language == "en":
        word_en_key, sentence_en_key, word_ro_key, sentence_ro_key, result_folder = (
            "3",
            "english",
            "gerunzius_and_lemmas",
            "7",
            "ingforms",
        )
    elif language == "ro":
        word_en_key, sentence_en_key, word_ro_key, sentence_ro_key, result_folder = (
            "adv_parts_and_lemmas",
            "7",
            "3",
            "romanian",
            "gerunzius",
        )

    matching_tuples = []
    all_errors = []
    for i, row in df.iterrows():
        if language == "en":
            word_en = row[word_en_key]
            word_ro = [x[0] for x in eval(row[word_ro_key])]
        elif language == "ro":
            word_en = [x[0] for x in eval(row[word_en_key])]
            word_ro = row[word_ro_key]
        sentence_en = row[sentence_en_key]
        sentence_ro = row[sentence_ro_key]
        truth_value, errors = is_match(
            word_en=word_en,
            sentence_en=sentence_en,
            word_ro=word_ro,
            sentence_ro=sentence_ro,
        )
        if truth_value or (not truth_value and not errors):
            matching_tuples.append((i, truth_value))
        if errors:
            for error in errors:
                print(f"{error.args[0]["word"]}, {error.args[0]["word"]}")
                all_errors.append(errors)

    result = (
        pd.DataFrame(
            matching_tuples,
            columns=["index", "part_matches_gerunziu"],
        ).set_index("index"),
        all_errors,
    )

    with open(result_folder + "/matching.pkl", "wb") as file:
        # Dump data with highest protocol for best performance
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)

    return result
