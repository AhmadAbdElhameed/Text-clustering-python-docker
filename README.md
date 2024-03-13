# Text-clustering-python-docker

### Text Clustering Application

This is a Flask application for performing text clustering. It accepts a text dataset file as input, clusters the text data, and provides the clustered data along with top keywords for each cluster.

## Requirements

- Python 3.11
- Flask
- pandas
- scikit-learn
- nltk

## Usage

1. Clone this repository to your local machine.

2. Install the required dependencies using pip:

    ```
    pip install -r requirements.txt
    ```

3. Run the Flask application:

    ```
    python app.py
    ```

4. Access the application in your web browser by visiting `http://localhost:5000`.

5. Upload a text dataset file and configure clustering parameters (optional).

6. Click on the "Cluster" button to perform text clustering.

7. Once clustering is completed, you can view the clustered data and download the output files.

## Docker

Alternatively, you can run the application using Docker. Follow these steps:

1. Build the Docker image:

    ```
    docker build -t my-flask-app .
    ```

2. Run the Docker container:

    ```
    docker run -d -p 5000:5000 -v /path/to/host/output:/usr/src/app/output my-flask-app
    ```

    Replace `/path/to/host/output` with the path to the directory where you want to save the output files on the host machine.

3. Access the application in your web browser by visiting `http://localhost:5000`.

## Output Files

The output files, including the clustered data and top keywords per cluster, will be saved in the specified output directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
