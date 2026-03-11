from gensim.models import KeyedVectors
from numpy import dot
from numpy.linalg import norm

ro = KeyedVectors.load_word2vec_format("wiki.ro.align.vec")
en = KeyedVectors.load_word2vec_format("wiki.en.align.vec")


def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def get_translation_score(ro_word, en_word):
    error = ""
    if ro_word not in ro:
        error = error + "rom not found"
        return -2
    if en_word not in en:
        error = error + " en not found"
        return -2
    if error != "":
        return -2
    return cosine(ro[ro_word], en[en_word])


def fix_ihat_ahat(word):
    word = word.replace("î", "â")
    if word[0] == "â":
        word = "î" + word[1:]
    if word[-1] == "â":
        word = word[:-1] + "î"
    return word


def fix_lines(word):
    return word.replace("ş", "ș").replace("ţ", "ț")


def get_best_match(word, lemma, candidate_words_and_lemmas, language_word):
    max_translation_score = -10
    best_match = None
    for candidate_word, candidate_lemma in candidate_words_and_lemmas:
        word = fix_lines(fix_ihat_ahat(word.lower()))
        lemma = fix_lines(fix_ihat_ahat(lemma.lower()))
        if language_word == "en":
            translation_score_word = get_translation_score(
                ro_word=candidate_word, en_word=word
            )
            translation_score_lemma = get_translation_score(
                ro_word=candidate_lemma, en_word=lemma
            )
        elif language_word == "ro":
            translation_score_word = get_translation_score(
                ro_word=word, en_word=candidate_word
            )
            translation_score_lemma = get_translation_score(
                ro_word=lemma, en_word=candidate_lemma
            )
        better_translation_score = max(translation_score_word, translation_score_lemma)
        if better_translation_score > max_translation_score:
            max_translation_score = better_translation_score
            best_match = (candidate_word, better_translation_score)
    return best_match
