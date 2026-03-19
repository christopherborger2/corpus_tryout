from simalign import SentenceAligner
import re
import pandas as pd
import pickle

aligner = SentenceAligner(model="bert")


def get_position_in_sentence(word, sentence_split):
    if not word in sentence_split:
        raise RuntimeError({"word": word, "sentence": sentence_split})
    return sentence_split.index(word)


def add_pronouns(word, sentence):
    if re.match(f".*{word}[^a-zăâîșț-].*", sentence):
        return word
    if not re.match(f".*({word}-?[a-zăâîșț]*-?[a-zăâîșț]*).*", sentence):
        print(f"add pronoun error with {word=}, {sentence=}")
    return re.match(f".*({word}-?[a-zăâîșț]*-?[a-zăâîșț]*).*", sentence).group(1)


def do_words_correspond(
    word_en, sentence_en_split, word_ro, sentence_ro_split, alignments
):
    position_ro = get_position_in_sentence(word_ro, sentence_ro_split)
    position_en = get_position_in_sentence(word_en, sentence_en_split)
    return (position_en, position_ro) in alignments


def split_sentence(sentence):
    splitting_chars = [
        "\n",
        ",",
        "‘",
        "’",
        ":",
        "\\!",
        "“",
        "”",
        ";",
        "'",
        "´",
        "~",
        '"',
        "…",
        "\\*",
        "•",
        "\\[",
        "\\]",
        "/",
        "`",
        "„",
        "«",
        "»",
        " ",
        "\\.",
        "\\?",
        "\\(",
        "\\)",
        "\\\\",
    ]
    return [
        remove_boundary_hyphens(w)
        for w in re.split("|".join(splitting_chars), sentence)
        if w != ""
    ]


def remove_boundary_hyphens(word):
    word_without_hyphens = re.match("-*([^-]?.*[^-])-*", word)
    if not word_without_hyphens:
        print(word)
        return ""
    return word_without_hyphens.group(1)


def is_match(word_en, sentence_en, word_ro, sentence_ro):

    sentence_en_split = split_sentence(sentence_en)
    sentence_ro_split = split_sentence(sentence_ro)

    alignments = aligner.get_word_aligns(sentence_en_split, sentence_ro_split)["mwmf"]

    errors = []

    if isinstance(word_en, list):
        word_ro = add_pronouns(word_ro, sentence_ro)
        for word in word_en:
            try:
                if do_words_correspond(
                    word, sentence_en_split, word_ro, sentence_ro_split, alignments
                ):
                    return True, errors
            except RuntimeError as e:
                print(f"Could not find word {e.args[0]["word"]}!")
                errors.append(e)
                continue
        return False, errors

    elif isinstance(word_ro, list):
        for word in word_ro:
            word = add_pronouns(word, sentence_ro)
            try:
                if do_words_correspond(
                    word_en, sentence_en_split, word, sentence_ro_split, alignments
                ):
                    return True, errors
            except RuntimeError as e:
                print(f"Could not find word {e.args[0]["word"]}!")
                errors.append(e)
                continue
        return False, errors

    else:
        word_ro = add_pronouns(word_ro, sentence_ro)
        try:
            return (
                do_words_correspond(
                    word_en, sentence_en_split, word_ro, sentence_ro_split, alignments
                ),
                errors,
            )
        except RuntimeError as e:
            print(f"Could not find word {e.args[0]["word"]}!")
            return False, errors


def match_adv_part_and_gerunzius(df, language, do_pickle=True):

    if language == "en":
        word_en_key, sentence_en_key, word_ro_key, sentence_ro_key, result_folder = (
            "3",
            "english",
            "gerunzius_and_lemmas",
            "romanian_new_orth",
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
    if do_pickle:
        with open(result_folder + "/matching.pkl", "wb") as file:
            # Dump data with highest protocol for best performance
            pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)

    return result
