import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import random
import spotipy as spt
from spotipy.oauth2 import SpotifyClientCredentials
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv('Stress_Dataset.csv')
categorical_columns = ['Nervousness', 'Unable to control', 'Worry', 'Trouble in Relaxation', 
                       'Restlessness', 'Irritability', 'Fear', 'Stress']

label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop('Stress', axis=1)
y = df['Stress']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=1000, random_state=42)
classifier.fit(X_train, y_train)

# Replace these with your actual Spotify API credentials
client_id = 'fe84d4f935064aa09b357a361475f6cc'
client_secret = '8f0349345905481a9885828a9de32c24'

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spt.Spotify(auth_manager=auth_manager)

def get_music_recommendations(genres, limit=5):
    songs = []
    for genre in genres:
        results = sp.search(q=f'{genre} ', type='track', limit=50)
        if results['tracks']['items']:
            for track in results['tracks']['items']:
                song_info = {
                    "name": track['name'],
                    "artist": track['artists'][0]['name'],
                    "album": track['album']['name'],
                    "image_url": track['album']['images'][0]['url'],
                    "preview_url": track['preview_url']
                }
                songs.append(song_info)
    if limit > len(songs):
        limit = len(songs)
    if limit <= 0:
        limit = 1
    return random.sample(songs, limit) if songs else ["No songs found for the given criteria."]

therapy_recommendations = {
    "No": {
        "music": ["pop", "classical", "acoustic"],
        "books": [
            {"title": "The Alchemist by Paulo Coelho", "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1SlUqJssoB2fTIdy5sSkCCkMS45oiN3ItKg&s"},
            {"title": "Remainders of him", "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTG8cp1JHWaqyIwYXp6kETSpBKs-WrREGolMQ&s"}
        ],
        "movies": [
            {"title": "Amélie", "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQvVKvq3Tg4U5p_JLNt-NZbgtwPHG7ee9jv_Q&s"},
            {"title": "The Secret Life of Walter Mitty", "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZ-CuvSBAsZ_CHXYTr5P7v1zIgkT_hmunQXQ&s"}
        ],
        "food": ["Dark chocolate", "Nuts and seeds"],
        # "exercise": ["Walking in nature", "Yoga"]
    },
    "Mild": {
        "music": ["jazz", "blues", "indie"],
        "books": [
            {"title": "Eleanor Oliphant Is Completely Fine by Gail Honeyman", "image_url": "https://www.blackwoolf.eu/wp-content/uploads/2020/08/977017.jpeg"},
            {"title": "The Rosie Project by Graeme Simsion", "image_url": "https://m.media-amazon.com/images/I/61a63kHpCdL._AC_UF1000,1000_QL80_.jpg"}
        ],
        "movies": [
            {"title": "Life of Pi", "image_url": "https://m.media-amazon.com/images/M/MV5BNTg2OTY2ODg5OF5BMl5BanBnXkFtZTcwODM5MTYxOA@@._V1_.jpg"},
            {"title": "Dear Zindagi", "image_url": "https://m.media-amazon.com/images/I/81JGFBE8ZaL._AC_UF1000,1000_QL80_.jpg"}
        ],
        "food": ["Pani Puri", "Green salads"],
        # "exercise": ["Moderate hiking", "Zumba"]
    },
    "Moderate": {
        "music": ["rock", "alternative", "hip-hop"],
        "books": [
            {"title": "The Road by Cormac McCarthy", "image_url": "https://upload.wikimedia.org/wikipedia/commons/2/27/The-road.jpg"},
            {"title": "Into the Wild by Jon Krakauer", "image_url": "https://images-eu.ssl-images-amazon.com/images/I/61A+LdmTESL._AC_UL210_SR210,210_.jpg"}
        ],
        "movies": [
            {"title": "Paramatma", "image_url": "https://m.media-amazon.com/images/M/MV5BYWEyYzEyMGMtZmUyYS00MTQ4LTgxNjgtMzljODdiMDJiZTNjXkEyXkFqcGdeQXVyODA4MDA0Mjg@._V1_.jpg"},
            {"title": "Chichore", "image_url": "https://content1.jdmagicbox.com/movies/centralized_121163742019_09_14_03_34_25_220.jpg"}
        ],
        "food": ["Comfort foods like mashed potatoes", "Stews and soups"],
        # "exercise": ["Running", "Swimming laps"]
    },
    "Severe": {
        "music": ["metal", "punk", "electronic"],
        "books": [
            {"title": "It starts with us", "image_url": "https://www.clankart.com/user-uploads/advert/It_starts_with_us__Colleen_Hoover1675533524594.jpg"},
            {"title": "It ends with us", "image_url": "https://images-eu.ssl-images-amazon.com/images/I/71Chcpw13lL._AC_UL210_SR210,210_.jpg"}
        ],
        "movies": [
            {"title": "3 idiots", "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ1XS4Oxu2UKvNAkWkZytpdymW-rSg2ORq5mQ&s"},
            {"title": "Yeh Jawaani Yeh Deewani", "image_url": "https://upload.wikimedia.org/wikipedia/en/thumb/1/15/Yeh_jawani_hai_deewani.jpg/220px-Yeh_jawani_hai_deewani.jpg"}
        ],
        "food": ["Ice Cream ", "Juice"],
        # "exercise": ["High-intensity interval training (HIIT)", "CrossFit"]
    }
}

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_data = {}
        for column in categorical_columns[:-1]:
            value = request.form[column]
            user_data[column] = value

        user_df = pd.DataFrame([user_data])
        for column in user_df.columns:
            le = label_encoders[column]
            user_df[column] = le.transform(user_df[column])

        prediction = classifier.predict(user_df)
        predicted_label = label_encoders['Stress'].inverse_transform(prediction)[0]

        recommendations = therapy_recommendations.get(predicted_label, {})
        genres = recommendations.get('music', [])
        music_recommendations = get_music_recommendations(genres)
        recommendations['music'] = music_recommendations

        return render_template('recommendations.html', 
                               stress_level=predicted_label, 
                               recommendations=recommendations)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
