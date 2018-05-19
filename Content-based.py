
# coding: utf-8

# In[2]:


import pandas as pd
import csv
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict

batch_size = 1000
itr = 10
wd = "/Users/XD/Downloads/"
app_info = defaultdict(list)
allAccuracies = []

with open(wd + "apps.tsv") as infile:
    reader = csv.reader(infile, delimiter="\t")
    next(reader) # burn the header
    for line in reader:
        (UserId, WindowID, Split, ApplicationDate, JobId) = line
        if WindowID != '1':
            break
        app_info[UserId].append(JobId)
        
for exp in range(itr):
    ds = pd.read_csv("/Users/XD/Downloads/splitjobs/jobs1.tsv",delimiter='\t',encoding='utf-8',
                     nrows=(exp+1)*batch_size, error_bad_lines=False)

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(ds['Description'].values.astype('U'))

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    results = {}

    for idx, row in ds.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-1000:-1]
        similar_items = [(cosine_similarities[idx][i], ds['JobID'][i]) for i in similar_indices]

        # First item is the item itself, so remove it.
        # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, item_id)
        results[row['JobID']] = similar_items[1:]
    
    maxId = max(results.keys())
    
    Accuracy = []
    for i in range(1, 11):
        accuracy = 0
        N = 0
        for userid in app_info:
            apps = [id for id in app_info[userid] if int(id) <= maxId]
            if len(apps) < 2:
                continue
            N += 1
            allRecs = defaultdict(float)
            randPick = random.randint(0, len(apps)-1)
            randJob = int(apps[randPick])
            apps.pop(randPick)
            recs = results[randJob][:10*i*len(apps)]
            for rec in recs:
                allRecs[str(rec[1])] += rec[0]

            accuracy += len(set(allRecs)&set(apps)) / float(len(apps))

        Accuracy.append(accuracy / float(N))
    print(Accuracy)
    allAccuracies.append(Accuracy)


# In[8]:


nTimesRecs = []
for i in range(1,11):
    nTimesRecs.append(10*i)
    
import matplotlib.pyplot as plt
plt.xlabel('recommendations/applications')
plt.ylabel('matching accuracy')
for i in range(itr):
    plt.plot(nTimesRecs, allAccuracies[i], label=str((i+1)*batch_size)+" jobs")
plt.legend()
plt.show()


# In[6]:


with open(wd + "apps.tsv") as infile:
    reader = csv.reader(infile, delimiter="\t")
    next(reader) # burn the header
    for line in reader:
        (UserId, WindowID, Split, ApplicationDate, JobId) = line
        if WindowID != '1':
            break
        app_info[UserId].append(JobId)

ds = pd.read_csv("/Users/XD/Downloads/splitjobs/jobs1.tsv",delimiter='\t',encoding='utf-8',
                 nrows=10000, error_bad_lines=False)

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['Description'].values.astype('U'))

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-1000:-1]
    similar_items = [(cosine_similarities[idx][i], ds['JobID'][i]) for i in similar_indices]

    # First item is the item itself, so remove it.
    # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, item_id)
    results[row['JobID']] = similar_items[1:]

maxId = max(results.keys())

print('done')


# In[10]:


Accuracy = []
for i in range(1, 16):
    accuracy = 0
    N = 0
    for userid in app_info:
        apps = [id for id in app_info[userid] if int(id) <= maxId]
        if len(apps) < 2*i:
            continue
        N += 1
        allRecs = defaultdict(float)
        for j in range(i):
            recs = results[int(apps[j])]
            for rec in recs:
                allRecs[str(rec[1])] += rec[0]
                
        recList = sorted(allRecs, key=allRecs.get, reverse=True)[:10*len(apps)]
        accuracy += len(set(recList)&set(apps)) / float(len(apps))

    Accuracy.append(accuracy / float(N))
print(Accuracy)

nRecBase = []
for i in range(1,16):
    nRecBase.append(i)
    
import matplotlib.pyplot as plt
plt.xlabel('number of known applications based on')
plt.ylabel('matched accuracy')
plt.plot(nRecBase, Accuracy)
plt.legend()
plt.show()

