import pickle
import streamlit as st
import requests

def fetch_poster(tmdbId):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(tmdbId)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def get_movie_detail(movie):
    movie_detail = movies[movies['title'] == movie]
    if not movie_detail.empty:
        movie_posters = fetch_poster(movie_detail['tmdbId'].values[0])
        movie_title = movie_detail['title'].values[0]
        movie_genres = movie_detail['genres'].values[0]
        movie_cast = movie_detail['cast'].values[0]
        movie_crew = movie_detail['crew'].values[0]
        movie_overview = movie_detail['overview'].values[0]
    else:
        return 'Movie Not Found'
    return movie_posters, movie_title, movie_genres, movie_cast, movie_crew, movie_overview
        
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        tmdbId = movies.iloc[i[0]].tmdbId
        recommended_movie_posters.append(fetch_poster(tmdbId))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommendations System Based on Searched Movie')
movies = pickle.load(open('artifacts/movie_list.pkl','rb'))
similarity = pickle.load(open('artifacts/similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Seacrh Movie'):
    posters, title, genres, cast, crew, overview = get_movie_detail(selected_movie)
    st.write("### Movie Search Result:")
    col1, col2 = st.columns([2,4])
    with col1:
        st.image(posters, width=200)
        st.markdown("<style>div[data-testid='stImage'] img {margin: 0;}</style>", unsafe_allow_html=True)
    with col2:
        st.write(f'# {title}')
        st.text(genres)
        st.text(f'Actors : {cast}')
        st.text(f'Director : {crew}')

    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    st.write("### Recommended Movies:")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(recommended_movie_posters[0])
        st.text(recommended_movie_names[0])
    with col2:
        st.image(recommended_movie_posters[1])
        st.text(recommended_movie_names[1])
    with col3:
        st.image(recommended_movie_posters[2])
        st.text(recommended_movie_names[2])
    with col4:
        st.image(recommended_movie_posters[3])
        st.text(recommended_movie_names[3])
    with col5:
        st.image(recommended_movie_posters[4])
        st.text(recommended_movie_names[4])

