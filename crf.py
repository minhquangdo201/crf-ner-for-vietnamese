import pandas as pd
import joblib
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score, flat_precision_score,flat_recall_score
from sklearn_crfsuite.metrics import flat_classification_report

# Đọc dữ liệu training
df = pd.read_csv('conll_train.csv', encoding = "utf-8")
df.head(10)
df.describe()
df['Tag'].unique()
df.isnull().sum()
df = df.fillna(method = 'ffill')

# Đọc dữ liệu test
test_file = pd.read_csv('conll_test.csv', encoding = "utf-8")
test_file.head(10)
test_file.describe()
test_file['Tag'].unique()
test_file.isnull().sum()
test_file = test_file.fillna(method = 'ffill')

# Chia dữ liệu theo câu
class sentence(object):
    def __init__(self, df):
        self.n_sent = 1
        self.df = df
        self.empty = False
        agg = lambda s : [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                       s['POS'].values.tolist(),
                                                       s['Tag'].values.tolist())]
        self.grouped = self.df.groupby("Sentence #").apply(agg)
        self.sentences = [s for s in self.grouped]
        
    def get_text(self):
        try:
            s = self.grouped['sentence {}'.format(self.n_sent)]
            self.n_sent +=1
            return s
        except:
            return None
        
# get train sentence    
getter = sentence(df)
sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
sent = getter.get_text()
sentences = getter.sentences

#get test sentence
tester = sentence(test_file)
tests = [" ".join([s[0] for s in sent]) for sent in tester.sentences]
test_sent = tester.get_text()
tests = tester.sentences

# Phân tích đặc trưng của từ
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
    }
    if i > 0:
        prev_word = sent[i-1][0]
        prev_word_lower = prev_word.lower()
        features.update({
            'prev_word.lower()': prev_word_lower,
            'prev_word[-3:]': prev_word[-3:],
            'prev_word[-2:]': prev_word[-2:],
            'prev_word.isupper()': prev_word.isupper(),
            'prev_word.istitle()': prev_word.istitle(),
            'prev_word.isdigit()': prev_word.isdigit()
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        next_word = sent[i+1][0]
        next_word_lower = next_word.lower()
        features.update({
            'next_word.lower()': next_word_lower,
            'next_word[-3:]': next_word[-3:],
            'next_word[-2:]': next_word[-2:],
            'next_word.isupper()': next_word.isupper(),
            'next_word.istitle()': next_word.istitle(),
            'next_word.isdigit()': next_word.isdigit()
        })
    else:
        features['EOS'] = True

    return features

# Lấy đặc trưng của các từ trong câu
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Lấy nhãn của các từ trong câu
def sent2labels(sent):
    return [label for token, postag, label in sent]

# def sent2tokens(sent):
#     return [token for token, postag, label in sent]

X_train = [sent2features(s) for s in sentences]
y_train = [sent2labels(s) for s in sentences]

X_test = [sent2features(s) for s in tests]
y_test = [sent2labels(s) for s in tests]


crf = CRF(algorithm = 'lbfgs',
        c1=1.0,
        c2=1e-3,
        max_iterations=200,
        all_possible_transitions=True,
        verbose=True)
crf.fit(X_train, y_train)

# # Lưu mô hình
joblib.dump(crf, 'crf_ner.pkl')

# #Tải mô hình
load_model = joblib.load('crf_ner.pkl')
# Dự đoán nhãn trên tập test
y_pred = load_model.predict(X_test)

# print('y TEST')
# print(y_test, end ='\n')
# print('y PRED')
# print(y_pred, end ='\n')


f1_score = flat_f1_score(y_true=y_test, y_pred=y_pred, average = 'weighted')
p = flat_precision_score(y_true=y_test, y_pred=y_pred, average = 'weighted')
r = flat_recall_score(y_true=y_test, y_pred=y_pred, average = 'weighted')

# labels = ['O','B-ADDRESS', 'I-ADDRESS', 'DATETIME', 'B-DATETIME-DATE','B-DATETIME-DATERANGE','B-DATETIME-DURATION','B-DATETIME-SET','B-DATETIME-TIME','B-DATETIME-TIMERANGE','B-EMAIL','B-EVENT','B-EVENT-CUL','B-EVENT-GAMESHOW','B-EVENT-NATURAL','B-EVENT-SPORT','B-IP','B-LOCATION','B-LOCATION-GEO','B-LOCATION-GPE','B-LOCATION-STRUC','B-MISCELLANEOUS','B-ORGANIZATION','B-ORGANIZATION-MED','B-ORGANIZATION-SPORT','B-ORGANIZATION-STOCK','B-PERSON','B-PERSONTYPE','B-PHONENUMBER','B-PRODUCT','B-PRODUCT-AWARD','B-PRODUCT-COM','B-PRODUCT-LEGAL','B-QUANTITY','B-QUANTITY-AGE','B-QUANTITY-CUR','B-QUANTITY-DIM','B-QUANTITY-NUM','B-QUANTITY-ORD','B-QUANTITY-PER','B-QUANTITY-TERM','B-SKILL','B-URL','I-DATETIME','I-DATETIME-DATE','I-DATETIME-DATERANGE','I-DATETIME-DURATION','I-DATETIME-SET','I-DATETIME-TIME','I-DATETIME-TIMERANGE','I-EMAIL','I-EVENT','I-EVENT-CUL','I-EVENT-GAMESHOW','I-EVENT-NATURAL','I-EVENT-SPORT','I-IP','I-LOCATION','I-LOCATION-GEO','I-LOCATION-GPE','I-LOCATION-STRUC','I-MISCELLANEOUS','I-ORGANIZATION','I-ORGANIZATION-MED','I-ORGANIZATION-SPORTS','I-ORGANIZATION-STOCK','I-PERSON','I-PERSONTYPE','I-PHONENUMBER','I-PRODUCT','I-PRODUCT-AWARD','I-PRODUCT-COM','I-PRODUCT-LEGAL','I-QUANTITY','I-QUANTITY-AGE','I-QUANTITY-CUR','I-QUANTITY-DIM','I-QUANTITY-NUM','I-QUANTITY-ORD','I-QUANTITY-PER','I-QUANTITY-TERM','I-SKILL','I-URL']
print("Precision: ",p, end='\n')
print("Recall: " ,r, end='\n')
print("F1: " ,f1_score, end='\n')

print(flat_classification_report(y_test, y_pred))