from flask import Flask, request, render_template, send_file, make_response
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk import PorterStemmer
import numpy as np
from io import BytesIO
import zipfile
import base64


app = Flask(__name__)

def clean_text(text):
    if text:
        clean = ' '.join(text.split())
        stem_text = [PorterStemmer().stem(word) for word in clean.split()]
        return ' '.join(stem_text)
    else:
        return text

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         file = request.files['file']
#         no_of_clusters = int(request.form['no_of_clusters'])
        
#         # Read uploaded file
#         data = pd.read_csv(file)
#         unstructure = request.form['col'] if 'col' in request.form else 'text'
#         data = data.fillna('NULL')
#         data['clean_sum'] = data[unstructure].apply(clean_text)
        
#         # Vectorize text
#         vectorizer = CountVectorizer(analyzer='word', stop_words='english')
#         counts = vectorizer.fit_transform(data['clean_sum'])
        
#         # Apply KMeans clustering
#         kmeans = KMeans(n_clusters=no_of_clusters)
#         data['cluster_sum'] = kmeans.fit_predict(counts)
        
#         # Extract top keywords for each cluster
#         cluster_centers = kmeans.cluster_centers_
#         feature_names = vectorizer.get_feature_names_out()
#         top_keywords = {}
#         for i in range(no_of_clusters):
#             cluster_center = cluster_centers[i]
#             top_keyword_indices = cluster_center.argsort()[-10:][::-1]  # Get indices of top 10 keywords
#             top_keywords[i] = [feature_names[idx] for idx in top_keyword_indices]
        
#         return render_template('result.html', data=data, top_keywords=top_keywords)
    
#     return render_template('index.html')


#  USING API ENDPOINT


# def index():
#     if request.method == 'POST':
#         file = request.files['file']
#         no_of_clusters = int(request.form['no_of_clusters'])
        
#         # Read uploaded file
#         data = pd.read_csv(file)
#         unstructure = request.form['col'] if 'col' in request.form else 'text'
#         data = data.fillna('NULL')
#         data['clean_sum'] = data[unstructure].apply(clean_text)
        
#         # Vectorize text
#         vectorizer = CountVectorizer(analyzer='word', stop_words='english')
#         counts = vectorizer.fit_transform(data['clean_sum'])
        
#         # Apply KMeans clustering
#         kmeans = KMeans(n_clusters=no_of_clusters)
#         data['cluster_sum'] = kmeans.fit_predict(counts)
        
#         # Extract top keywords for each cluster
#         cluster_centers = kmeans.cluster_centers_
#         feature_names = vectorizer.get_feature_names_out()
#         top_keywords = {}
#         for i in range(no_of_clusters):
#             cluster_center = cluster_centers[i]
#             top_keyword_indices = cluster_center.argsort()[-10:][::-1]  # Get indices of top 10 keywords
#             top_keywords[i] = [feature_names[idx] for idx in top_keyword_indices]
        
#         # Saving the clustered data to a CSV file
#         output_filename = 'clustered_data.csv'
#         data.to_csv(output_filename, index=False)
        
#         # Saving top keywords for each cluster to a DataFrame
#         top_keywords_df = pd.DataFrame(top_keywords)
#         top_keywords_filename = 'top_keywords_per_cluster.csv'
#         top_keywords_df.to_csv(top_keywords_filename, index=False)
        
#         # Create an in-memory zip file
#         zip_buffer = BytesIO()
#         with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zipf:
#             zipf.write(output_filename)
#             zipf.write(top_keywords_filename)
        
#         # Go back to the beginning of the buffer
#         zip_buffer.seek(0)
        
#         # Create response with HTML and attached zip file
#         response = make_response(zip_buffer.getvalue())
#         response.headers['Content-Type'] = 'application/zip'
#         response.headers['Content-Disposition'] = 'attachment; filename=clustering_results.zip'
#         return response
    
#     return render_template('index.html')


# Define custom filter for base64 encoding
@app.template_filter('b64encode')
def b64encode(data):
    return base64.b64encode(data.getvalue()).decode('utf-8')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        no_of_clusters = int(request.form['no_of_clusters'])
        
        # Read uploaded file
        data = pd.read_csv(file)
        unstructure = request.form['col'] if 'col' in request.form else 'text'
        data = data.fillna('NULL')
        data['clean_sum'] = data[unstructure].apply(clean_text)
        
        # Vectorize text
        vectorizer = CountVectorizer(analyzer='word', stop_words='english')
        counts = vectorizer.fit_transform(data['clean_sum'])
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=no_of_clusters)
        data['cluster_sum'] = kmeans.fit_predict(counts)
        
        # Extract top keywords for each cluster
        cluster_centers = kmeans.cluster_centers_
        feature_names = vectorizer.get_feature_names_out()
        top_keywords = {}
        for i in range(no_of_clusters):
            cluster_center = cluster_centers[i]
            top_keyword_indices = cluster_center.argsort()[-10:][::-1]  # Get indices of top 10 keywords
            top_keywords[i] = [feature_names[idx] for idx in top_keyword_indices]
        
        # Saving the clustered data to a CSV file
        output_filename = 'clustered_data.csv'
        data.to_csv(output_filename, index=False)
        
        # Saving top keywords for each cluster to a DataFrame
        top_keywords_df = pd.DataFrame(top_keywords)
        top_keywords_filename = 'top_keywords_per_cluster.csv'
        top_keywords_df.to_csv(top_keywords_filename, index=False)
        
        # Create an in-memory zip file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zipf:
            zipf.write(output_filename)
            zipf.write(top_keywords_filename)
        
        # Go back to the beginning of the buffer
        zip_buffer.seek(0)
        
        # Pass the data and top keywords to the result template
        return render_template('result.html', data=data, top_keywords=top_keywords, zip_buffer=zip_buffer)
    
    return render_template('index.html')


@app.route('/download', methods=['POST'])
def download():
    if request.method == 'POST':
        zip_buffer = request.form['zip_buffer']
        zip_buffer = BytesIO(base64.b64decode(zip_buffer))  # Decode base64 data
        zip_buffer.seek(0)
        return send_file(zip_buffer, as_attachment=True, mimetype='application/zip', download_name='clustering_results.zip')
    else:
        return "Method Not Allowed", 405



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
