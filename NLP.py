#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk')
import nltk
nltk.download('punkt')


# In[2]:


import nltk
nltk.download('punkt_tab')


# In[3]:


import nltk
from nltk.tokenize import sent_tokenize

text = "I come to Rusia. It is a very beautiful place."

sentences = sent_tokenize(text)
print("Result:")
for i, sent in enumerate(sentences, 1):
    print(f"sentence{i}: {sent}")


# In[4]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

text = "I come to Rusia. It is a very beautiful place."

sentences = sent_tokenize(text)
print("sentence's result:")
for i, sent in enumerate(sentences, 1):
    print(f"sentence{i}: {sent}")

print("\nword's result:")
for i, sent in enumerate(sentences, 1):
    words = word_tokenize(sent)
    print(f"sentence{i}'s words: {words}")


# In[ ]:




