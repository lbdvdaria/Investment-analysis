# Textual analysis and machine learning.
# 1. Calculate the similarity score (cosine similarity) between quarterly reports from t-1 to t and plot your results
# 2. Use LM negative word list to test the tone of quarterly reports from your chosen
# company and plot your results
# 3. Draw word clouds of quarterly reports from your chosen company and briefly
# explain the figures
# 4. Assess whether the daily return in the filing date can be explained by the
# similarity score or the negative tone

import pandas as pd
import numpy as np
import os
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
### Download 10-K and 1-Q reports for JNJ (Johnson & Johnson - chosen company) for the period from January 1, 2010 to December 31, 2020
from sec_edgar_downloader import Downloader

dl = Downloader()
dl.get("10-Q", "JNJ", after="2010-01-01", before="2020-12-31")
dl.get("10-K", "JNJ", after="2010-01-01", before="2020-12-31")


#  Step 1: Calculation of Cosine Similarity
def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('txt'):
                file_list.append(os.path.join(root, file))
    return file_list


# Definition of the function to  find key words
def find_keyword(file_path):
    keywords = dfstem_gen.set_index('gen_words')['count'].to_dict()
    file_name = os.path.basename(file_path)
    content_string = get_string(file_path)
    porterstem_filter = clean_text(content_string)
    for word in porterstem_filter:
        if word.upper() in keywords:
            keywords[word.upper()] += 1
    return file_name, keywords


# Definition of the function to read lines from a txt file
def get_non_empty_lines(filePath):
    lines = open(filePath, errors='ignore').read().splitlines()
    non_empty_lines = list()
    for line in lines:
        if line.strip():
            line = line.upper()
            non_empty_lines.append(line)
    return non_empty_lines


# Definition of the function to get financial period end date
def get_date(non_empty_lines):
    datadate = ''
    for line in non_empty_lines:
        match = re.search('FILED AS OF DATE', line)
        if match:
            datadate = re.sub('[^0-9]', '', line)
    return datadate


# Definition of the function to read txt as string
def get_string(filePath):
    content_string = open(filePath).read().replace('\n', ' ')
    return content_string


# Definition of the function to remove the stop words from the content
def clean_text(content_string):
    content_string = re.sub(r'\<.*?\>', ' ', content_string)
    content_string = re.sub('[0-9]', ' ', content_string)
    content_string = re.sub(r'[^\w\s]', ' ', content_string)
    content_string = content_string.lower()
    word_tokens = word_tokenize(content_string)
    nltk_stopwords = list(stopwords.words('english'))
    filter = []
    for word in word_tokens:
        if word not in nltk_stopwords:
            filter.append(word)
    porter = PorterStemmer()
    porterstem_filter = [porter.stem(word) for word in filter]
    return porterstem_filter


# Step 2: Data Preparation

file = open(r'C:\Users\PC\coding\finance\T3_data_files\generalword_list.txt', 'r')
general_word = file.readlines()
general_wordlist = []
for word in general_word:
    tmp = word.strip('\n')
    general_wordlist.append(tmp)

# Import stopword list from NLTK package
nltk_stopwords = list(stopwords.words('english'))

# Removal of stopwords from General Words list
general_nostop = []
for word in general_wordlist:
    if word not in nltk_stopwords:
        general_nostop.append(word)

porter = PorterStemmer()
stem_generalwords = [porter.stem(word) for word in general_nostop]

# Removal of duplicated stems and transferring the list into dataframe
dfstem_gen = pd.DataFrame(stem_generalwords)
dfstem_gen = dfstem_gen.drop_duplicates().reset_index(drop=True)

zero_list = [0] * len(dfstem_gen)
dfstem_gen['count'] = zero_list
dfstem_gen.columns = ['gen_words', 'count']
dfstem_gen['gen_words'] = dfstem_gen['gen_words'].str.upper()

# Creation of Initial list
data = []
cik_list = []
date_list = []
file_index = []
cum = []
keywords_count = []
percentage = []


# Definition of the function to get file list
for path in get_file_list(os.getcwd()):
    file_name, word_distribution = find_keyword(path)
    non_empty_lines = get_non_empty_lines(path)
    date = get_date(non_empty_lines)
    file_index.append(file_name)
    date_list.append(date)
    for k in word_distribution:
        data.append(word_distribution[k])

# Function output
output = np.array(data).reshape(int(len(data) / len(dfstem_gen)), len(dfstem_gen))
final_output = pd.DataFrame({'date': date_list})
general_vector = []
for i in range(len(output)):
    general_vector.append(list(output[i]))
final_output['general_words'] = general_vector

#  Rearranging dates into Year and Month
final_output['date'] = final_output['date'].astype(str)
final_output['date'] = pd.to_datetime(final_output['date'])
final_output['year'] = final_output['date'].dt.year
final_output['month'] = final_output['date'].dt.month
final_output['quarter'] = final_output['date'].dt.quarter

final_output = final_output.iloc[1:, :].reset_index(drop=True)
final_output.to_csv(r'C:\Users\PC\coding\finance\T3_data_files\general_word_distribution.csv')


# Step 3: Calculation of Cosine Similarity

# Definition of the function to calculate Cosine similarity
def cosine_sim(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

# Creation of a new working file and temporary variable
df = final_output  # the values were sorted in the previous step by Year and Month
df = df.sort_values(by=['quarter', 'year']).reset_index(drop=True)
temp = np.array(df.general_words.tolist())

# Calculation of the number of documents
Number_of_documents = [0 for _ in range(len(dfstem_gen))]
for i in range(len(dfstem_gen)):
    for count in temp[:, i]:
        if count > 0:
            Number_of_documents[i] += 1

# Transferring into array format
No_documents_w = np.array(Number_of_documents)

# Checking how many words never exist in all the documents
print(len(No_documents_w[No_documents_w == 0]))

# Calculation of IDF
# No_documents_w = (temp != 0).sum(0)  # No. of documents that include word w
idf_list = len(temp) / No_documents_w  # len(temp) = No. of documents in sample
# Some words may never exist in all documents,
# in this case, we can change the equation to : len(temp)/(1+No_documents_w)
idf_list = np.log(idf_list)
where_are_inf = np.isinf(idf_list)
idf_list[where_are_inf] = 0  # If no documents include word w, then we set the idf weight equal to zero

# Weighted Cosine Similarity
tfidf_list = []
for i in df.general_words:
    tfidf = (i * idf_list) / sum(i)
    tfidf_list.append(tfidf)

df['tfidf'] = tfidf_list

# Definition of the function
for i in range(len(df)):
    tmp = np.array(df.general_words.tolist())
    No_documents_w_i = (tmp != 0).sum(0)
    idf_list_i = len(tmp) / (1 + No_documents_w_i)
    idf_list_i = np.log(idf_list_i)

    tfidf_list_i = []
    for j in df.general_words:
        tfidf_i = (j * idf_list_i) / sum(j)
        tfidf_list_i.append(tfidf_i)

    df['tfidf_i'] = tfidf_list_i

# Calculation of the Cosine Similarity based on simple count and tfidf weight between the quarters for the company

cos_sim_list = []
tfidf_sim_list = []
quarter_list = []
year_list = []

for i in range(len(df) - 1):
    cos_sim = ''
    year = ''
    if df.loc[i + 1, 'quarter'] == df.loc[i, 'quarter'] and (df.loc[i + 1, 'year'] - df.loc[i, 'year']) == 1:
        x = df.loc[i + 1, 'general_words']
        y = df.loc[i, 'general_words']
        x1 = df.loc[i + 1, 'tfidf_i']
        y1 = df.loc[i, 'tfidf_i']

        cos_sim = cosine_sim(np.array(x), np.array(y))
        tfidf_sim = cosine_sim(np.array(x1), np.array(y1))

        year = df.loc[i + 1, 'year']
        quarter = df.loc[i + 1, 'quarter']

    if cos_sim != '':
        cos_sim_list.append(cos_sim)
        tfidf_sim_list.append(tfidf_sim)
        quarter_list.append(quarter)
        year_list.append(year)

result = pd.DataFrame(
    {'year': year_list, 'quarter': quarter_list, 'cos_sim': cos_sim_list, 'tfidf_sim': tfidf_sim_list})
print(result)

result.to_csv('Cosine Similarity_Output.csv')

# Step 2: Sentiment Analysis: Negative Tone

def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('txt'):
                file_list.append(os.path.join(root, file))
    return file_list

# Definition of the function to  find negative words
def find_keyword(file_path):
    keywords = dfstem_Neg.set_index('Negwords')['count'].to_dict()
    file_name = os.path.basename(file_path)
    content_string = get_string(file_path)
    porterstem_filter = clean_text(content_string)
    total_words = len(porterstem_filter)
    content_after_stem = ' '.join(porterstem_filter)

    for word in porterstem_filter:
        if word.upper() in keywords:
            keywords[word.upper()] += 1
    return file_name, keywords, total_words, content_after_stem

# Definition of the function to read lines from a txt file
def get_non_empty_lines(filePath):
    lines = open(filePath).read().splitlines()
    non_empty_lines = list()
    for line in lines:
        if line.strip():
            line = line.upper()
            non_empty_lines.append(line)
    return non_empty_lines

# Definition of the function to get financial period end date
def get_date(non_empty_lines):
    datadate = ''
    for line in non_empty_lines:
        match = re.search('FILED AS OF DATE', line)
        if match:
            datadate = re.sub('[^0-9]', '', line)
    return datadate

# Definition of the function to read txt as string
def get_string(filePath):
    content_string = open(filePath).read().replace('\n', ' ')
    return content_string

# Definition of the function to remove the stop words from the content
def clean_text(content_string):  # initial prep
    content_string = re.sub(r'\<.*?\>', ' ', content_string)
    content_string = re.sub('[0-9]', ' ', content_string)
    content_string = re.sub(r'[^\w\s]', ' ', content_string)
    content_string = content_string.lower()
    word_tokens = word_tokenize(content_string)
    nltk_stopwords = list(stopwords.words('english'))

    filter = []
    for word in word_tokens:
        if word not in nltk_stopwords:
            filter.append(word)
    porter = PorterStemmer()
    porterstem_filter = [porter.stem(word) for word in filter]
    return porterstem_filter


# Importing Negative Words List
FinNeg = pd.read_excel(r'C:\Users\PC\coding\finance\T3_data_files\FinNeg.xlsx')

temp = FinNeg['Negative_words'].tolist()
porter = PorterStemmer()
stem_Negwords = [porter.stem(word) for word in temp]
stem_Negwords = list(set(stem_Negwords))

dfstem_Neg = pd.DataFrame(stem_Negwords)
zero_list = [0] * len(stem_Negwords)
dfstem_Neg['1'] = zero_list
dfstem_Neg.columns = ['Negwords', 'count']
dfstem_Neg['Negwords'] = dfstem_Neg['Negwords'].str.upper()

column = ['date', 'negwords', 'cum', 'proportion']
ind = column + stem_Negwords

# Creation of Initial list
data = []
date_list = []
file_index = []
cum = []
keywords_count = []
percentage = []
corpus = []

# Definition of the function to get file list
for path in get_file_list(os.getcwd()):
    file_name, word_contribution, total_words, content_after_stem = find_keyword(path)
    non_empty_lines = get_non_empty_lines(path)
    date = get_date(non_empty_lines)
    date_list.append(date)
    cum.append(total_words)
    corpus.append(content_after_stem)

    for k in word_contribution:
        data.append(word_contribution[k])

# Function output
output = np.array(data).reshape(int(len(data) / len(stem_Negwords)), len(stem_Negwords))
keywords_count = np.sum(output, axis=1)
percentage = [a / b for a, b in zip(keywords_count, cum)]
percentage = [i * 100 for i in percentage]
output = np.c_[date_list, keywords_count, cum, percentage, output]
pd_output = pd.DataFrame(output, columns=ind)
final_output = pd_output.iloc[4:, :].reset_index(drop=True)
final_output = final_output[['date', 'negwords', 'cum', 'proportion']]

# Rearraging date
final_output = final_output.sort_values(['date']).reset_index(drop=True)
final_output['date'] = final_output['date'].astype(str)
final_output['date'] = pd.to_datetime(final_output['date'])
final_output['year'] = final_output['date'].dt.year
final_output['month'] = final_output['date'].dt.month
final_output['quarter'] = final_output['date'].dt.quarter

final_output.to_csv('Negative_Words_output.csv')


#  Step 3: Word Cloud
# Definition of the function to clean text
def clean_text(content_string):
    content_string = re.sub(r'\<.*?\>', ' ', content_string)
    content_string = re.sub('[0-9]', ' ', content_string)
    content_string = re.sub(r'[^\w\s]', ' ', content_string)
    content_string = content_string.lower()

    word_tokens = word_tokenize(content_string)
    nltk_stopwords = list(stopwords.words('english'))

    filter = []
    for word in word_tokens:
        if word not in nltk_stopwords:
            filter.append(word)

    porter = PorterStemmer()
    porterstem_filter = [porter.stem(word) for word in filter]


# Open General List file
file = open(r'C:\Users\PC\coding\finance\T3_data_files/generalword_list.txt', 'r')

# Definition of the function to get string
def get_string(filePath):
    content_string = open(filePath).read().replace('\n', ' ')
    return content_string

# Removal of the '\n' at the end of each line
general_word = file.readlines()
general_wordlist = []
for word in general_word:
    tmp = word.strip('\n')
    general_wordlist.append(tmp)

# Import of stopword list from NLTK package
nltk_stopwords = list(stopwords.words('english'))

# Removal of stopwords
general_nostop = []
for word in general_wordlist:
    if word not in nltk_stopwords:
        general_nostop.append(word)

# Generation of a stemmed general word list without stopwords, 5804 in total
porter = PorterStemmer()
stem_generalwords = [porter.stem(word) for word in general_nostop]

# Genrate Word Cloud of report released in Q2'2011
content_Q2_11 = get_string(r'C:\Users\PC\coding\finance\T3_data_files/Q22011.txt')
cleaned_content = clean_text(content_Q2_11)

g = ''.join(content_Q2_11)
wordcloud = WordCloud(background_color="white").generate(g)
plt.figure()

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud.to_file("Q22011.png")

# Genrate Word Cloud of report released in Q4'2010
content_Q4_10 = get_string(r'C:\Users\PC\coding\finance\T3_data_files/Q42010.txt')
cleaned_content = clean_text(content_Q4_10)

g = ''.join(content_Q4_10)
wordcloud = WordCloud(background_color="white").generate(g)
plt.figure()

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud.to_file("Q42010.png")

# Genrate Word Cloud of report released in Q2'2012
content_Q2_12 = get_string(r'C:\Users\PC\coding\finance\T3_data_files/Q22012.txt')
cleaned_content = clean_text(content_Q2_12)

g = ''.join(content_Q2_12)
wordcloud = WordCloud(background_color="white").generate(g)
plt.figure()

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud.to_file("Q22012.png")

# Genrate Word Cloud of report released in Q2'2019
content_Q2_19 = get_string(r'C:\Users\PC\coding\finance\T3_data_files/Q22019.txt')
cleaned_content = clean_text(content_Q2_19)

g = ''.join(content_Q2_19)
wordcloud = WordCloud(background_color="white").generate(g)
plt.figure()

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud.to_file("Q22019.png")

# Genrate Word Cloud of report released in Q2'2020
content_Q2_20 = get_string(r'C:\Users\PC\coding\finance\T3_data_files/Q22020.txt')
cleaned_content = clean_text(content_Q2_20)

g = ''.join(content_Q2_20)
wordcloud = WordCloud(background_color="white").generate(g)
plt.figure()

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud.to_file("Q22020.png")

# Genrate Word Cloud of report released in Q2'2020
content_Q4_20 = get_string(r'C:\Users\PC\coding\finance\T3_data_files/Q42020.txt')
cleaned_content = clean_text(content_Q4_20)

g = ''.join(content_Q4_20)
wordcloud = WordCloud(background_color="white").generate(g)
plt.figure()

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud.to_file("Q42020.png")