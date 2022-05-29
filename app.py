from flask import Flask
import flask
import csv
from flask import Flask, render_template, request
import difflib
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bs4 as bs
import urllib.request
import pickle
import numpy as np


from tmdbv3api import TMDb
tmdb = TMDb()
tmdb.api_key = '55c4c871d39f738e4d9b897fcec8db01'
app = Flask(__name__)


app = flask.Flask(__name__, template_folder='templates')

movies=pd.read_csv('./model/tmdb_5000_movies.csv')
credits=pd.read_csv('./model/tmdb_5000_credits.csv')

movies=movies.merge(credits,on='title')

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew','release_date']]

#checking for the duplicate values
movies.dropna(inplace=True)

ast.literal_eval
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)


def convert2(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if(counter!=4):
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert2)

#fetching the director name from the column
def fetch_d(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if(i['job'] == 'Director'):
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_d)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])


movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


moviesdf = movies[['movie_id','title','tags']]


moviesdf['tags'] = moviesdf['tags'].apply(lambda x:" ".join(x))


moviesdf['tags'] = moviesdf['tags'].apply(lambda x:x.lower())
moviesdf['title'] = moviesdf['title'].apply(lambda x:x.lower())


ps=PorterStemmer()


#steming the keywords
def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


moviesdf['tags'] = moviesdf['tags'].apply(stem)

moviesdf.to_csv (r'D:\Movie Recommendator\moviesdf.csv', index=None)

cv=CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(moviesdf['tags']).toarray()

similarity = cosine_similarity(vectors)

#moviesdf[moviesdf['title'] == 'Aliens'].index[0] #masking


def recommend(movie):
    movie_index = moviesdf[moviesdf['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(moviesdf.iloc[i[0]].title)


app = Flask(__name__)



clf = pickle.load(open('movies_dict.pkl', 'rb'))
vectorizer = pickle.load(open('movies_dict.pkl','rb'))

def create_similarity():
    data = pd.read_csv('./model/moviesdf.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['tags'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape

    except:
        data, similarity = create_similarity()
    if m not in data['title'].unique():
        print(m)
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:9] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['title'][a])
        return l
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('./model/moviesdf.csv')
    return list(data['title'].str.capitalize())



@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    # vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}     

    # passing all the data to the html file
    return render_template('recommend.html',title=title,poster=poster,overview=overview,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,casts=casts,cast_details=cast_details)


if __name__ == "__main__":
    app.run(debug=True, port=7000)