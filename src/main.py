import pandas as pd

import spacy
import re

# Dados
df_twitter = pd.read_json(r"C:\Users\claud\ufcg\reproducao\src\dataset_twitter.json", encoding='utf-8')
df_buscape = pd.read_json(r"C:\Users\claud\ufcg\reproducao\src\dataset_buscape.json", encoding='utf-8')
lexicon = set(open(r"C:\Users\claud\ufcg\reproducao\src\portuguese_lexicon.csv").read().split())


# Carregue o modelo de português do spaCy
nlp = spacy.load("pt_core_news_sm")

# LEXICON_ORIENTATION = load_lexicon(lexicon)


# 3) Detecção de preferência:
def get_word_orientation(word):
    return 1;


def get_related_entity(text, keyword, ent1, ent2) -> str:
    doc = nlp(text)
    keyword_token = None
    for token in doc:
        if token.text.lower() == keyword.lower():
            keyword_token = token
            break

    if not keyword_token:
        return ent1

    # Verifica se ent1 ou ent2 está mais próximo da palavra-chave
    distances = {}
    for ent in [ent1, ent2]:
        for token in doc:
            if ent.lower() in token.text.lower():
                distances[ent] = abs(token.i - keyword_token.i)
                break

    if not distances:
        return ent1

    return min(distances, key=distances.get)


DECREMENT_WORDS = ["quase", "um pouco", "levemente"]

def contains_decrement_expression(keyword, text):
    for word in DECREMENT_WORDS:
        pattern = r"\b" + re.escape(word) + r"\s+" + re.escape(keyword) + r"\b"
        if re.search(pattern, text.lower()):
            return True
    return False


def contains_negation(keyword, text):
    pattern = r"(não|nunca|jamais)\s+(.*?)\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, text.lower()) is not None


def get_feature_orientation(text):
    return 1


# Algoritmo 1: Non-Equal-Gradable sentences
def get_preference(text, keyword, aspect, ent1, ent2):
    word_orientation = get_word_orientation(keyword)
    related_entity = get_related_entity(text, keyword, ent1, ent2)

    if contains_decrement_expression(keyword, text):
        word_orientation *= -1

    if contains_negation(keyword, text):
        word_orientation *= -1

    if aspect is not None:
        aspectOrientation = get_feature_orientation(aspect)
        word_orientation *= aspectOrientation

    if word_orientation > 0:
        return related_entity
    else:
        return ent2 if related_entity == ent1 else ent1


# Algoritmo 2: Superlative sentences
def is_superlative_sentence_preferred(text, keyword, aspect):
    word_orientation = get_word_orientation(keyword)

    if contains_decrement_expression(keyword, text):
        word_orientation *= -1

    if contains_negation(keyword, text):
        word_orientation *= -1

    if aspect is not None:
        aspect_orientation = get_feature_orientation(aspect)
        word_orientation *= aspect_orientation

    return word_orientation > 0


y_true_superlative = []
y_pred_superlative = []

superlative_labels = []
non_equal_gradable_sentences = []

y_true_non_equal_gradable = []
y_pred_non_equal_gradable = []


for sentence in df_buscape.to_dict(orient='records'):
    for label in sentence.get('labels'):
        label_type = label.get('type')
        if label_type == '3':
            y_true_superlative.append(label.get('preferred_entity'))

            result = is_superlative_sentence_preferred(sentence.get('text'),
                                                       label.get('keyword'),
                                                       label.get('aspect'))
            if result:
                y_pred_superlative.append(label.get('entity_s1'))
            else:
                y_pred_superlative.append(label.get('entity_s2'))

        elif label_type == '1':
            y_true_non_equal_gradable.append(label.get('preferred_entity'))
            result = get_preference(sentence.get('text'),
                                    label.get('keyword'),
                                    label.get('aspect'),
                                    label.get('entity_s1'),
                                    label.get('entity_s2'))

            y_pred_non_equal_gradable.append(result)


print(y_true_superlative)
print(y_pred_superlative)


from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true_superlative, y_pred_superlative, average='macro')
recall = recall_score(y_true_superlative, y_pred_superlative, average='macro')
f1 = f1_score(y_true_superlative, y_pred_superlative, average='macro')

print(precision)
print(recall)
print(f1)