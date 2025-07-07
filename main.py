#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd


# In[22]:


df = pd.read_csv('spam.csv', encoding = "ISO-8859-1")


# In[23]:


df.sample(5)


# In[24]:


df.shape


# ## 1. DATA CLEANING

# In[25]:


df.info()

 


# In[26]:


# drop last 3 cols
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True)


# In[27]:


df.sample(5)


# In[28]:


df.rename(columns={'v1': 'target', 'v2':'text'}, inplace=True)
df.sample(5)


# In[29]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[30]:


df['target'] = encoder.fit_transform(df['target'])


# In[31]:


df.head()


# In[32]:


df.isnull().sum()


# In[33]:


df.duplicated().sum()


# In[34]:


df = df.drop_duplicates(keep='first')


# In[35]:


df.duplicated().sum()


# In[36]:


df.shape


# ## 2. EDA

# In[37]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()


# In[38]:


# Data is imbalanced
import nltk
import os
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)
nltk.download('stopwords')


# In[39]:


df['num_characters'] = df['text'].apply(len)


# In[40]:


df.head()


# In[41]:


df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[42]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[43]:


df.head()


# In[44]:


df[['num_characters','num_words', 'num_sentences']].describe()


# In[45]:


# good(ham) msgs
df[df['target'] == 0][['num_characters','num_words', 'num_sentences']].describe()


# In[46]:


# spam msgs
df[df['target'] == 1][['num_characters','num_words', 'num_sentences']].describe()


# In[47]:


import seaborn as sns


# In[48]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters']) #type: ignore
sns.histplot(df[df['target']==1]['num_characters'], color='red') #type: ignore


# In[49]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words']) #type: ignore
sns.histplot(df[df['target']==1]['num_words'], color='red') #type: ignore


# In[50]:


sns.pairplot(df, hue='target')


# In[51]:


sns.heatmap(df.corr(numeric_only=True), annot=True)


# ## 3. Data Preprocessing

# In[52]:


import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# removing special characters and lowering by splitting each word (word tokenize)
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
        
    return " ".join(y)


# In[53]:


transform_text("I loved the YT lectures on Machine Learning. How about you?")


# In[54]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[55]:


df.head()


# In[56]:


from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')


# In[57]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[58]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[59]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[60]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[61]:


spam = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for words in msg.split():
        spam.append(words)
        


# In[62]:


len(spam)


# In[63]:


from collections import Counter
counts = pd.DataFrame(Counter(spam).most_common(30))
sns.barplot(x=counts[0], y=counts[1])
plt.xticks(rotation='vertical')
plt.show()


# In[64]:


ham = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for words in msg.split():
        ham.append(words)


# In[65]:


len(ham)


# In[66]:


from collections import Counter
counts = pd.DataFrame(Counter(ham).most_common(30))
sns.barplot(x=counts[0], y=counts[1])
plt.xticks(rotation='vertical')
plt.show()


# ## 4. Model Building

# In[67]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[68]:


X = tfidf.fit_transform(df['transformed_text']).toarray() #type: ignore


# In[69]:


X.shape


# In[70]:


y= df['target'].values


# In[71]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=2)


# In[72]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[73]:


gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))


# In[74]:


mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))


# In[75]:


bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))


# In[76]:


# tfidf --> MNB: https://github.com/campusx-official/sms-spam-classifier/blob/main/sms-spam-detection.ipynb

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
# need to install some runtime
# from xgboost import XGBClassifier


# In[77]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)


# In[78]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
}


# In[79]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[80]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[81]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[82]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[83]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm") #type: ignore


# In[84]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[85]:


# improve the model --> bro also had a scaled data version, did not copy over


# In[86]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[87]:


new_df = performance_df.merge(temp_df, on='Algorithm')


# In[88]:


svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[89]:


# combines multiple
voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[90]:


voting.fit(X_train,y_train)


# In[91]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[92]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[93]:


from sklearn.ensemble import StackingClassifier


# In[94]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[95]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[96]:


# all above was was for testing we actually have to use the fitted mnb module:
mnb.fit(X_train, y_train)


# In[97]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[98]:


# 
# for msg in df[df['target'] == 1]['transformed_text'].tolist():
#     print(msg)

