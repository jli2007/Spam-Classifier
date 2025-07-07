#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv('spam.csv', encoding = "ISO-8859-1")


# In[4]:


df.sample(5)


# In[5]:


df.shape


# ## 1. DATA CLEANING

# In[6]:


df.info()

 


# In[7]:


# drop last 3 cols
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True)


# In[8]:


df.sample(5)


# In[9]:


df.rename(columns={'v1': 'target', 'v2':'text'}, inplace=True)
df.sample(5)


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[11]:


df['target'] = encoder.fit_transform(df['target'])


# In[12]:


df.head()


# In[13]:


df.isnull().sum()


# In[14]:


df.duplicated().sum()


# In[15]:


df = df.drop_duplicates(keep='first')


# In[16]:


df.duplicated().sum()


# In[17]:


df.shape


# ## 2. EDA

# In[18]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()


# In[19]:


# Data is imbalanced
import nltk

# already downloaded:

# nltk.download('punkt', download_dir="nltk_data")
# nltk.download('stopwords', download_dir="nltk_data")

# tells code to look in nltk_folder directory for nltk data
nltk.data.path.append('nltk_data')

# In[20]:


df['num_characters'] = df['text'].apply(len)


# In[21]:


df.head()


# In[22]:


df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[23]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[24]:


df.head()


# In[25]:


df[['num_characters','num_words', 'num_sentences']].describe()


# In[26]:


# good(ham) msgs
df[df['target'] == 0][['num_characters','num_words', 'num_sentences']].describe()


# In[27]:


# spam msgs
df[df['target'] == 1][['num_characters','num_words', 'num_sentences']].describe()


# In[28]:


import seaborn as sns


# In[29]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters']) # type: ignore
sns.histplot(df[df['target']==1]['num_characters'], color='red') # type: ignore


# In[30]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words']) # type: ignore
sns.histplot(df[df['target']==1]['num_words'], color='red') # type: ignore


# In[31]:


sns.pairplot(df, hue='target')


# In[32]:


sns.heatmap(df.corr(numeric_only=True), annot=True)


# ## 3. Data Preprocessing

# In[33]:


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


# In[34]:


transform_text("I loved the YT lectures on Machine Learning. How about you?")


# In[35]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[36]:


df.head()


# In[37]:


from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')


# In[38]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[39]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[40]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[41]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[42]:


spam = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for words in msg.split():
        spam.append(words)
        


# In[43]:


len(spam)


# In[44]:


from collections import Counter
counts = pd.DataFrame(Counter(spam).most_common(30))
sns.barplot(x=counts[0], y=counts[1])
plt.xticks(rotation='vertical')
plt.show()


# In[45]:


ham = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for words in msg.split():
        ham.append(words)


# In[46]:


len(ham)


# In[47]:


from collections import Counter
counts = pd.DataFrame(Counter(ham).most_common(30))
sns.barplot(x=counts[0], y=counts[1])
plt.xticks(rotation='vertical')
plt.show()


# ## 4. Model Building

# In[48]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[49]:


X = tfidf.fit_transform(df['transformed_text']).toarray() # type: ignore


# In[50]:


X.shape


# In[51]:


y= df['target'].values


# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=2)


# In[53]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[54]:


gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))


# In[55]:


mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))


# In[56]:


bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))


# In[57]:


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


# In[58]:


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


# In[59]:


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


# In[60]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[61]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[62]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[63]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[64]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm") # type: ignore


# In[65]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[66]:


# improve the model --> bro also had a scaled data version, did not copy over


# In[67]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[68]:


new_df = performance_df.merge(temp_df, on='Algorithm')


# In[69]:


svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[70]:


# combines multiple
voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[71]:


voting.fit(X_train,y_train)


# In[72]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[73]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[74]:


from sklearn.ensemble import StackingClassifier


# In[75]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[76]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[ ]:


# all above was was for testing we actually have to use the fitted mnb module:
mnb.fit(X_train, y_train)


# In[79]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[82]:


# 
# for msg in df[df['target'] == 1]['transformed_text'].tolist():
#     print(msg)

print("script finished")