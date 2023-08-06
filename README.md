# Movie Recommendation System using Natural Language Processing (NLP)

## Overview
This project is a Movie Recommendation System that utilizes Natural Language Processing (NLP) techniques to suggest similar movies based on user input. The system processes movie data, including movie descriptions, genres, keywords, cast, and crew, to create a comprehensive set of tags for each movie. These tags are then transformed using stemming and vectorization to facilitate similarity comparison. Finally, the cosine similarity measure is employed to recommend movies with the highest similarity to the user's input.

## Project Setup
1. Install the required libraries: Make sure you have the necessary Python libraries like NumPy, Pandas, scikit-learn, and nltk installed. You can install them using pip: `pip install numpy pandas scikit-learn nltk`

2. Data Collection: Obtain the movie dataset files, `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`, containing movie details and credits information, respectively.
3. Dataset_used:-
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

4. Data Preprocessing: The provided Python script handles data preprocessing, including merging datasets, converting nested JSON fields, and creating tags for each movie.

5. Stemming and Vectorization: Stemming reduces words to their root form, and CountVectorizer converts the text data into a numerical representation to enable cosine similarity computation.

6. Movie Recommendation: The project uses cosine similarity to recommend similar movies based on user input.

## Usage
1. Import Required Libraries: Ensure you have the necessary libraries imported at the beginning of your code.

2. Load and Preprocess Data: Load the movie dataset using Pandas and preprocess it using the provided script.

3. Movie Recommendation: To recommend movies similar to a specific movie, call the `recommend()` function with the movie title as input.

## Example Code
python
# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

# Step 2: Load and Preprocess Data
# ... (Include code to load and preprocess the movie dataset)

# Step 3: Movie Recommendation
def recommend(movie):
    # ... (Include the code for the recommend() function)

# Example Usage:
recommend('Avatar')  # This will recommend similar movies to 'Avatar'

# Step 4: Save the Processed Data and Similarity Matrix
import pickle 
# Save the processed movie dataset
pickle.dump(new_df, open('movies.pkl', 'wb'))

# Save the movie dataset as a dictionary for easy access
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))

# Save the cosine similarity matrix
pickle.dump(similarity, open('similarity.pkl', 'wb'))


*Note:* Before running the project, ensure that you have the required dataset files in the correct format and modify the file paths accordingly. The project assumes that you have already installed the necessary libraries and have Python installed on your system.

## Getting Started
To get started with this project, follow these steps:

1. Clone the repository to your local machine using `git clone <repository_url>`.

2. Install the required libraries by running `pip install -r requirements.txt`.

3. Download the movie dataset files, `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`, and place them in the same directory as the Python script.

4. Run the Python script to preprocess the data and create the movie recommendation system.

5. Now you can use the `recommend()` function to find similar movies based on user input.

Feel free to explore the code, modify it to suit your preferences, and have fun discovering new movies!

## Acknowledgments
Special thanks to the creators of the TMDb dataset for providing the movie data used in this project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions or suggestions regarding this project, feel free to contact me at [saurabhsinghcse3112@gmail.com]

