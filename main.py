from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
import spacy
nlp = spacy.load('en_core_web_sm')

feature_names = [
    'index',
    'text',
    'pos',
    'dep',
    'begin',
    'end',
    'index_2',
    'is_upper',
    'is_lower',
    'head_text',
    'head_pos',
    'head_dep',
    'is_aux',  # ADD   
    'num_verbs',  # ADD
    'entity',
    'relation'
]


def _parse_data(path, names):
    result = []
    for doc in open(path).read().split("\n\n"):
        doc_repr = []
        tokens = doc.split("\n")
        for token in tokens:
            values = token.strip().split(" ")
            if len(values) == 0:
                continue
            doc_repr.append(dict(zip(names, values)))
        result.append(doc_repr)
    return result


train_data = _parse_data('./data/train.conll', feature_names)
test_data = _parse_data('./data/test.conll', feature_names)


def word2features(sent, i):
    word = sent[i]
    features = [
        'word.lower=' + word['text'].lower(),  # word.lower(),
        'word.isaux=' + word['is_aux'],
        'word.num_verbs=' + word['num_verbs'],
        'word[-3:]=' + word['text'][-3:],
        # 'word[-2:]=' + word['text'][-2:],
        'word.isupper=%s' % word['text'].isupper(),
        'word.istitle=%s' % word['text'].istitle(),
        'word.isdigit=%s' % word['text'].isdigit(),
        'postag=' + word['pos'],
        'postag[:2]=' + word['pos'][:2],
        'dep=' + word['dep'],
    # 'head_dep=' + word['head_dep'],
    ]
    if i > 0:
        word1 = sent[i - 1]
        features.extend([
            '-1:word.lower=' + word1['text'].lower(),
            '-1:word.istitle=%s' % word1['text'].istitle(),
            '-1:word.isupper=%s' % word1['text'].isupper(),
            '-1:postag=' + word1['pos'],
            '-1:postag[:2]=' + word1['pos'][:2],
            '-1:dep=' + word1['dep'],
             '-1:head_dep=' + word1['head_dep'],
        ])
    if i > 1:
        word1 = sent[i - 2]
        features.extend([
            '-2:word.lower=' + word1['text'].lower(),
            '-2:word.istitle=%s' % word1['text'].istitle(),
            '-2:word.isupper=%s' % word1['text'].isupper(),
            '-2:postag=' + word1['pos'],
            '-2:postag[:2]=' + word1['pos'][:2],
            '-2:dep=' + word1['dep'],
            '-2:head_dep=' + word1['head_dep'],
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.extend([
            '+1:word.lower=' + word1['text'].lower(),
            '+1:word.istitle=%s' % word1['text'].istitle(),
            '+1:word.isupper=%s' % word1['text'].isupper(),
            '+1:postag=' + word1['pos'],
            '+1:postag[:2]=' + word1['pos'][:2],
            '+1:dep=' + word1['dep'],
            '+1:head_dep=' + word1['head_dep'],
        ])

    if i < len(sent) - 2:
        word1 = sent[i + 2]
        features.extend([
            '+2:word.lower=' + word1['text'].lower(),
            '+2:word.istitle=%s' % word1['text'].istitle(),
            '+2:word.isupper=%s' % word1['text'].isupper(),
            '+2:postag=' + word1['pos'],
            '+2:postag[:2]=' + word1['pos'][:2],
            '+2:dep=' + word1['dep'],
            '+2:head_dep=' + word1['head_dep'],
        ])
    else:
        features.append('EOS')

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent, ent=True):
    label = 'entity' if ent else 'relation'
    return [w[label] for w in sent]


def sent2tokens(sent):
    return [w['text'] for w in sent]


X_train = [sent2features(s) for s in train_data]
y_train = [sent2labels(s) for s in train_data]

X_test = [sent2features(s) for s in test_data]
y_test = [sent2labels(s) for s in test_data]


trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 100,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('eswcSWO.crfsuite')

tagger = pycrfsuite.Tagger()
tagger.open('eswcSWO.crfsuite')

example_sent = train_data[4389]
print(' '.join(sent2tokens(example_sent)), end='\n\n')

print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
print("Correct:  ", ' '.join(sent2labels(example_sent)))


#cq = "Which pizza toppings can I use to process this fork?"
#cq_repr = []
#doc = nlp(cq)
#for token in doc:
#    cq_repr.append([str(token.i), token.text, token.pos_, token.dep_,
#                    str(token.idx), str(token.idx + len(token)),
#                    str(token.i), str(token.is_upper),
#                    str(token.is_lower), token.head.text,
#                    token.head.pos_, token.head.dep_,
#                    _get_iobs(token, annotations, "E"),
#                    _get_discontinuous_iobs(token, annotations, "R")])


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )


y_pred = [tagger.tag(xseq) for xseq in X_test]
print(bio_classification_report(y_test, y_pred))
