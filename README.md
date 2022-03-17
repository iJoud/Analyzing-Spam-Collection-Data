# Analyzing Spam Collection Data
Built spam detector model with Naïve Bayes and NLP techniques, using Scikit-learn, and NLTK on a dataset of spam messages.

This is an assignment I've completed on simplilearn Data Science with Python Course.

## Followed Approach: 
1. Eliminate the punctuation marks and stopwords from the given dataset
2. Apply feature extraction using **bag of words**, and **Tf-idf transformer**
3. **Train ML model**; Detect Spam with Naïve Bayes model

**Finally**, I've tested model predictions against actual responses e.g. in following code snippet I choose randomly the 84th message from the dataset, 
apply bag of words and  Tf-idf transformer. Then, in the print statement print predicted data using the trained spam detector model followed by the actual response. 

````Python
# 1 transform the message using bag of words
msg = df_spam_collections['messages'][84]
bag_of_words_for_msg = bag_of_words_transformer.transform([msg])
# 2 transform the message using tfidf 
tfidf = tfidf_transformer.transform(bag_of_words_for_msg)
print(f'Message 84\npred = {spam_detect_model.predict(tfidf)}'
+f'\nactual = {df_spam_collections.response[84]}')
````
