import streamlit as st
import pickle
import pandas as pd
import requests
import base64

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=fcfe9d00147e628720bf64c74ab19642&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id


        recommended_movies.append(movies.iloc[i[0]].title)
        # fetch poster from API
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies,recommended_movies_posters

movies_dict = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))

# Set the Streamlit app theme with the desired background color
st.set_page_config(layout="wide", page_title="Movie Recommender System", page_icon=":movie_camera:")

# Replace the below hex value with your desired background color code
bg_color = "#780000"
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background-color: {bg_color};
        }}
        .sidebar .sidebar-content {{
            background-color: {bg_color};
        }}
        div.stButton > button {{
            background-color: #780000; /* Change this to the desired color of the "Recommend" button */
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
'Choose a movie',
movies['title'].values)


if st.button('Recommend'):
    names,posters = recommend(selected_movie_name)

    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])