#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:31:36 2024

@author: ahmad
"""

from nltk import PorterStemmer
from flask import Flask , request , make_response , send_file , render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
from io import BytesIO
import zipfile

app = Flask(__name__)



def clean_text(text):
    if text:
        clean = ' '.join(text.split())
        stem_text = [PorterStemmer().stem(word) for word in clean.split()]
        
        return ' '.join(stem_text)
    else:
        return text
 

@app.route('/cluster',methods = ['POST'])
def cluster():
    data = pd.read_csv(request.files['dataset']) 
    
    unstructure = 'text'
    
    if 'col' in request.args:
        unstructure = request.args.get('col')
    
    no_of_clusters = 2
    
    if 'no_of_clusters' in request.args:
        no_of_clusters = int(request.args.get('no_of_clusters'))
    
    data = data.fillna('NULL')
    
    data['clean_sum'] = data[unstructure].apply(clean_text)
    
    vectorizer = CountVectorizer(analyzer='word', stop_words='english')
    counts = vectorizer.fit_transform(data['clean_sum'])
    
    kmeans = KMeans(n_clusters=no_of_clusters)
    data['cluster_sum'] = kmeans.fit_predict(counts)
    
    data = data.drop(['clean_sum'], axis=1)
    
    # Saving the clustered data to a new CSV file
    output_filename = 'clustered_data.csv'
    data.to_csv(output_filename, index=False)
    
    
    # Extracting top keywords for each cluster
    cluster_centers = kmeans.cluster_centers_
    feature_names = vectorizer.get_feature_names_out()
    top_keywords = {}
    for i in range(no_of_clusters):
        cluster_center = cluster_centers[i]
        top_keyword_indices = cluster_center.argsort()[-10:][::-1]  # Get indices of top 10 keywords
        top_keywords[i] = [feature_names[idx] for idx in top_keyword_indices]
    
    # Saving top keywords for each cluster to a DataFrame
    top_keywords_df = pd.DataFrame(top_keywords)
    top_keywords_filename = 'top_keywords_per_cluster.csv'
    top_keywords_df.to_csv(top_keywords_filename, index=False)
    
    # Create an in-memory zip file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zipf:
        zipf.write(output_filename)
        zipf.write(top_keywords_filename)
    
    # Create a response containing the zip file
    response = make_response(zip_buffer.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=clustered_data_and_top_keywords.zip'
    response.headers['Content-Type'] = 'application/zip'
    
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    
    


# def clean_text(text):
#     if text:
#         clean = ' '.join(text.split())
#         stem_text = [PorterStemmer().stem(word) for word in clean.split()]
#         return ' '.join(stem_text)
#     else:
#         return text

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/cluster', methods=['POST'])
# def cluster():
#     uploaded_file = request.files['file']
#     if uploaded_file.filename == '':
#         return render_template('index.html', error='No file selected.')

#     unstructured_column = request.form.get('column')
#     no_of_clusters = int(request.form.get('clusters', 2))

#     data = pd.read_csv(uploaded_file)
#     data = data.fillna('NULL')

#     vectorizer = CountVectorizer(analyzer='word', stop_words='english')
#     counts = vectorizer.fit_transform(data[unstructured_column].apply(clean_text))

#     kmeans = KMeans(n_clusters=no_of_clusters)
#     data['cluster'] = kmeans.fit_predict(counts)

#     # Extracting top keywords for each cluster
#     feature_names = vectorizer.get_feature_names_out()
#     top_keywords = {}
#     for i in range(no_of_clusters):
#         cluster_center = kmeans.cluster_centers_[i]
#         top_keyword_indices = cluster_center.argsort()[-10:][::-1]  # Get indices of top 10 keywords
#         top_keywords[i] = [feature_names[idx] for idx in top_keyword_indices]

#     return render_template('index.html', data=data.to_html(), top_keywords=top_keywords)

# if __name__ == '__main__':
#     app.run(debug=True)
    