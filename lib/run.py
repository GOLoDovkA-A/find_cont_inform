fp_train = '/task-for-hiring-data/train.csv'
fp_val = '/task-for-hiring-data/val.csv'
fp_test = '/task-for-hiring-data/test_data.csv'

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import datetime, time
import re
import nltk

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import csv

cfg = dict(parse_dates=['datetime_submitted'])

train = pd.read_csv(fp_train, **cfg)
val = pd.read_csv(fp_val, **cfg)

data = pd.concat([train, val], axis=0)
data_test = pd.read_csv(fp_test, **cfg)

def create_features(df):
    
    hour = df['datetime_submitted'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df["description_len"] = df.description.apply(lambda x: len(x))
    df["description_quant_of_7"] = df.description.apply(lambda x: x.count("7"))
    # df["description_quant_of_8"] = df.description.apply(lambda x: x.count("8"))
    # df["description_quant_of_9"] = df.description.apply(lambda x: x.count("9"))
    # df["description_quant_of_numb"] = df.description.apply(lambda x: len(re.findall(r"\d", x)))
    # df["description_qunt_of_unique"] = data.description.apply(lambda x: len(set(x)))
    df["title_quant_of_numb"] = df.title.apply(lambda x: len(re.findall(r"\d", x)))
    df["title_len"] = df.title.apply(lambda x: len(x))
    df["title_qunt_of_unique"] = df.title.apply(lambda x: len(set(x)))
    df["title_quant_of_7"] = df.title.apply(lambda x: x.count("7"))
    df["title_quant_of_8"] = df.title.apply(lambda x: x.count("8"))
    df["title_quant_of_9"] = df.title.apply(lambda x: x.count("9"))


    regex_phone = r"([8-9]\d{7,10})|(\+7\d{7,10})|((\d.){8,11})|(\+7 \d{3})|(8[(-]\d{3})|(89 )|([8-9] \d)"
    regex_mess = r"(vk.com)|(Discord)|(What's app)|(Whats app)|(Whatsapp)|(вотсап)|(вацап)|(viber)|(вайбер)"
    regex_email = r"(http)|(@mail)|(@yandex)|(@yahoo)|(@gmail)|(@ya)|(@list)|(@bk)|(@outlook)"
    # df["regex_phone_true"] = [1 if i==True else 0 for i in df.description.str. \
    #                     contains(regex_phone).fillna(False)]
    df["regex_email_true"] = [1 if i==True else 0 for i in df.description.str. \
                        contains(regex_email).fillna(False)]
    # df["regex_mess_true"] = [1 if i==True else 0 for i in df.description.str. \
    #                     contains(regex_phone).fillna(False)]
    df["regex_phone_true_title"] = [1 if i==True else 0 for i in df.title.str. \
                        contains(regex_phone).fillna(False)]
    df["regex_email_true_title"] = [1 if i==True else 0 for i in df.title.str. \
                        contains(regex_email).fillna(False)]
    df["regex_mess_true_title"] = [1 if i==True else 0 for i in df.title.str. \
                        contains(regex_phone).fillna(False)]
    return df

def preproc(df):
    
    df["description_title"] = df.title + df.description
    df = df.drop(["title", "description","datetime_submitted"], axis=1)
    
    df.price[df.price > df.price.quantile(0.95)] = np.nan
    df.price = df.groupby(by="subcategory"). \
                    transform(lambda x: x.fillna(x.median())).price

    trash_regex = r"[\\\"\n\/\!\?\#\=\№\%\:\^\&\*\~\`\$\;\-\_\`\'\.\t]"
    df.description_title = df.description_title.apply(lambda x:
                                            re.sub(trash_regex, " ", x))
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
                           "]+", flags=re.UNICODE)
    df.description_title = df.description_title. \
                        apply(lambda x: emoji_pattern.sub(r'', x))
    
    return df

data = data.pipe(create_features)
data_test = data_test.pipe(create_features)
data = data.pipe(preproc)
data_test = data_test.pipe(preproc)

le_subcat = LabelEncoder()
data.subcategory = le_subcat.fit_transform(data.subcategory)
le_cat = LabelEncoder()
data.category = le_cat.fit_transform(data.category)
le_region = LabelEncoder()
data.region = le_region.fit_transform(data.region)
le_city = LabelEncoder()

data.city = le_city.fit_transform(data.city)
data_test.subcategory = le_subcat.transform(data_test.subcategory)
data_test.category = le_cat.transform(data_test.category)
data_test.region = le_region.transform(data_test.region)
data_test.city = le_city.transform(data_test.city)

russian_stopwords = ['и','в','во','не','что','он','на','я','с','со','как','а','то','все','она','так','его','но','да','ты','к','у','же','вы','за','бы','по','только','ее','мне','было','вот','от','меня','еще','нет','о','из','ему','теперь','когда','даже','ну','вдруг','ли','если','уже','или','ни','быть','был','него','до','вас','нибудь','опять','уж','вам','ведь','там','потом','себя','ничего','ей','может','они','тут','где','есть','надо','ней','для','мы','тебя','их','чем','была','сам','чтоб','без','будто','чего','раз','тоже','себе','под','будет','ж','тогда','кто','этот','того','потому','этого','какой','совсем','ним','здесь','этом','один','почти','мой','тем','чтобы','нее','сейчас','были','куда','зачем','всех','никогда','можно','при','наконец','два','об','другой','хоть','после','над','больше','тот','через','эти','нас','про','всего','них','какая','много','разве','три','эту','моя','впрочем','хорошо','свою','этой','перед','иногда','лучше','чуть','том','нельзя','такой','им','более','всегда','конечно','всю','между']

TF_IDF = TfidfVectorizer(max_df=0.5, min_df=0.0001,
                     stop_words=russian_stopwords)
TF_IDF.fit(data.description_title)
description_tf_idf_data = TF_IDF.transform(data.description_title)

svd = TruncatedSVD(n_components=40)
description_tf_idf_data_svd = pd.DataFrame(svd.fit_transform(description_tf_idf_data))

description_tf_idf_data_test = TF_IDF.transform(data_test.description_title)
description_tf_idf_data_test_svd = pd.DataFrame(svd.transform(description_tf_idf_data_test))

X = pd.concat([data, description_tf_idf_data_svd], axis=1, ignore_index=True)
X_test = pd.concat([data_test, description_tf_idf_data_test_svd], axis=1, ignore_index=True)

IF = IsolationForest(n_jobs=-1, n_estimators=140)
IF.fit(X.drop(["description_title"], axis=1))
X_weights = IF.predict(X.drop(["description_title"], axis=1))
X_weights = pd.Series(np.where(X_weights == 1, 1, 0))

clf_rf = RandomForestClassifier(n_estimators = 130, n_jobs=-1)
clf_rf.fit(X.drop(["description_title", "is_bad"], axis=1), X["is_bad"], sample_weight=X_weights)

prediction = clf_rf.predict_proba(X_test.drop(["description_title", "is_bad"], axis=1))[:,1]

prediction = pd.Series(prediction)
index = pd.Series(range(0,len(prediction)))
target_prediction = pd.DataFrame({"index":index, "prediction":prediction})

target_prediction.to_csv("/task-for-hiring-data/target_prediction.csv", sep='\t', encoding='utf-8')
