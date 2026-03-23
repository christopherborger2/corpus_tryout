import pickle
import stanza
import os
import re

nlp_en = stanza.Pipeline("en")
nlp_ro = stanza.Pipeline("ro")

SEP_en = "This is a splitting sentence."
SEP_ro = "Asta este o propoziție."

orthography_pairs = dict([("sînt", "sunt"), ("sîntem", "suntem")])

prefixes = [
    "bine",
    "ne",
    "re",
    "pre",
    "de",
    "des",
    "rău",
    "prea",
    "sub",
    "supra",
    "micro",
    "nemai",
    "auto",
    "anti",
    "agro",
]


def convert_word_to_new_orthography(word):
    if word in orthography_pairs.keys():
        return orthography_pairs[word]
    for prefix in prefixes:
        if word.startswith(prefix):
            return word
    word = re.sub(r"(?<=\w)î(?=\w)", "â", word)
    word = re.sub(r"\bâ(?=\w)", "î", word)
    word = re.sub(r"\bÂ(?=\w)", "Î", word)
    word = re.sub(r"(?<=\w)â\b", "î", word)
    word = re.sub(r"(?<=\w)Â\b", "Î", word)
    return word


def convert_sentence_to_new_orthography(sentence):
    # Split into words, punctuation, and whitespace
    tokens = re.findall(r"\s+|\w+|[^\w\s]", sentence, re.UNICODE)
    converted_tokens = [
        convert_word_to_new_orthography(t) if re.match(r"\w", t, re.UNICODE) else t
        for t in tokens
    ]
    # Rejoin preserving whitespace
    return "".join(converted_tokens)


def get_doc(df_slice, column, model, sep):
    merged = f" {sep} ".join(df_slice[column].astype(str)) + sep
    return model(merged)


def pickle_doc(doc, df_slice, name, folder):
    min_index = df_slice.index[0]
    max_index = df_slice.index[-1]
    filename = f"{folder}/parsed_slice_{name}_{min_index}_{max_index}.pkl"
    with open(filename, "wb") as file:
        # Dump data with highest protocol for best performance
        pickle.dump(doc, file, protocol=pickle.HIGHEST_PROTOCOL)


def parse_in_chunks(df, language, colname_text, folder, chunk_size, startwith_chunk=1):

    if language == "en":
        sep, nlp, pickle_name = SEP_en, nlp_en, "en"
    elif language == "ro":
        sep, nlp, pickle_name = SEP_ro, nlp_ro, "ro"

    i = 0 + (startwith_chunk - 1) * chunk_size

    while True:
        if i + chunk_size <= len(df):
            print(f"Parsing chunk {i} to {i + chunk_size - 1}")
            df_slice = df.iloc[i : i + chunk_size]
            doc = get_doc(df_slice, colname_text, nlp, sep)
            pickle_doc(doc, df_slice, pickle_name, folder)
            i = i + chunk_size
        else:
            print(f"Parsing chunk {i} to {len(df) - 1}")
            df_slice = df.iloc[i : len(df)]
            doc = get_doc(df_slice, colname_text, nlp, sep)
            pickle_doc(doc, df_slice, pickle_name, folder)
            break


def get_and_pickle_doc(chunk, colname_text, nlp, sep, pickle_name, folder):
    """
    Wrapper of get_doc and pickle_doc in order to run this combination in parallel
    """
    doc = get_doc(chunk, colname_text, nlp, sep)
    pickle_doc(doc, chunk, pickle_name, folder)


def split_feats(feats):
    feats_list = feats.split("|")
    feats_dict = dict()
    for feat in feats_list:
        feat_name_value = feat.split("=")
        feats_dict[feat_name_value[0]] = feat_name_value[1]
    return feats_dict


def is_head_of_be(word, sentence):
    for w in sentence.words:
        if w.lemma == "be" and w.head == word.id:
            return True
    return False


def is_preposition(word):
    return word.upos == "SCONJ" and word.xpos == "IN" and word.deprel == "mark"


def is_head_of_preposition(word, sentence):
    for w in sentence.words:
        if w.head == word.id and is_preposition(w):
            return True
    return False


def sentences_contain_adv_part(matched_word, sentences):
    for sentence in sentences:
        for word in sentence.words:
            if matched_word and not word.text == matched_word:
                continue
            if not word.feats:
                continue
            if is_adv_part(word, sentence):
                print(word)
                return (True, word.lemma)
    return (False, None)


def sentences_contain_gerunziu(matched_word, sentences):
    print(matched_word)
    for sentence in sentences:
        for word in sentence.words:
            if matched_word and not word.text == matched_word:
                continue
            if not word.feats:
                continue
            if is_gerunziu(word):
                print(word)
                return (True, word.lemma)
    return (False, None)


def get_word_by_id(id, sentence):
    return sentence.words[id - 1]


def is_adv_part(word, sentence):

    word_features = split_feats(word.feats)
    return (
        word.text[-3:] == "ing"
        and word.upos == "VERB"
        and (word.deprel == "advcl" or word.deprel == "conj")
        and word_features["VerbForm"] == "Part"
        and not is_head_of_preposition(word, sentence)
        and not is_head_of_be(word, sentence)  # avoid progressive
        and not (
            word.deprel == "conj"
            and is_head_of_preposition(
                get_word_by_id(
                    word.head,
                    sentence,
                ),
                sentence,
            )  # avoid "running" in "in swimming and running"
        )
        and not (
            word.deprel == "conj" and get_word_by_id(word.head, sentence).upos == "ADJ"
        )
    )


def is_gerunziu(word):
    word_features = split_feats(word.feats)
    return "VerbForm" in word_features.keys() and word_features["VerbForm"] == "Ger"


def get_adv_part_in_sentences(sentences):
    adv_parts = []

    for sentence in sentences:
        for word in sentence.words:
            if not word.feats:
                continue
            if is_adv_part(word, sentence):
                adv_parts.append((word.text, word.lemma))
    return adv_parts


def get_gerunzius_in_sentences(sentences):
    gerunzius = []

    for sentence in sentences:
        for word in sentence.words:
            if not word.feats:
                continue
            if is_gerunziu(word):
                gerunzius.append((word.text, word.lemma))

    return gerunzius


def remove_sep(text, sep):
    return text.replace(sep, "")


def split_doc(doc, language):

    if language == "en":
        sep, nlp = SEP_en, nlp_en
    elif language == "ro":
        sep, nlp = SEP_ro, nlp_ro

    row_text = []
    current_sentences = []
    row = 0

    for sentence in doc.sentences:
        if sentence.text == sep:  # sentences correctly split
            row_text.append((row, current_sentences))
            current_sentences = []
            row = row + 1
        elif sentence.text.endswith(sep):  # sep left at end of a sentence
            clean_sentence = remove_sep(sentence.text, sep)
            clean_doc = nlp(clean_sentence)
            current_sentences = current_sentences + clean_doc.sentences
            row_text.append((row, current_sentences))
            current_sentences = []
            row = row + 1
        elif sentence.text.startswith(sep):  # sep left at the start of a sentence
            row_text.append((row, current_sentences))
            current_sentences = []
            row = row + 1
            clean_sentence = remove_sep(sentence.text, sep)
            clean_doc = nlp(clean_sentence)
            current_sentences = current_sentences + clean_doc.sentences
        elif sep in sentence.text:
            sentence_parts = sentence.text.split(sep)
            clean_doc_1 = nlp(sentence_parts[0])
            current_sentences = current_sentences + clean_doc_1.sentences
            row_text.append((row, current_sentences))
            current_sentences = []
            row = row + 1
            clean_doc_2 = nlp(sentence_parts[1])
            current_sentences = current_sentences + clean_doc_2.sentences
        else:
            current_sentences.append(sentence)

    return dict(row_text)


def load_pickled_data(folder, language):
    parsed_dicts = {}
    for _, _, files in os.walk(folder):
        for file in files:
            print(file)
            match = re.match(f".*_{language}_(.*)_.*pkl", file)
            if match:
                doc = pickle.load(open(folder + "/" + file, "rb"))
                parsed_dicts[int(match.group(1))] = split_doc(
                    doc=doc, language=language
                )

    new_parsed_dicts = {}
    for k in parsed_dicts.keys():
        new_parsed_dicts[k] = dict(
            zip(
                [i + k for i in parsed_dicts[k].keys()],
                parsed_dicts[k].values(),
            )
        )

    parsed_dict = {}
    for d in new_parsed_dicts.values():
        parsed_dict.update(d)

    with open(folder + "/parsed_dict_" + language + ".pkl", "wb") as file:
        # Dump data with highest protocol for best performance
        pickle.dump(parsed_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    return parsed_dict
