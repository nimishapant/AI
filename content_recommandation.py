from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample movie data
import pandas as pd

data = {
    'movie_id': list(range(1, 51)),
    'title': [
        "The Matrix", "John Wick", "Toy Story", "The Lord of the Rings", "The Lion King",
        "Inception", "Interstellar", "Gladiator", "Titanic", "Avatar",
        "The Dark Knight", "Pulp Fiction", "Fight Club", "Forrest Gump", "The Shawshank Redemption",
        "The Godfather", "The Avengers", "Jurassic Park", "Star Wars", "Back to the Future",
        "The Silence of the Lambs", "Saving Private Ryan", "Schindler's List", "Braveheart", "Casablanca",
        "Goodfellas", "The Departed", "Whiplash", "La La Land", "The Prestige",
        "The Social Network", "Guardians of the Galaxy", "Deadpool", "The Hunger Games", "Mad Max: Fury Road",
        "Black Panther", "Wonder Woman", "Spider-Man", "Iron Man", "Doctor Strange",
        "The Incredibles", "Finding Nemo", "Monsters, Inc.", "Up", "Inside Out",
        "Coco", "Frozen", "Moana", "Zootopia", "Ratatouille"
    ],
    'description': [
        "A computer hacker discovers reality is a simulation controlled by machines.",
        "An ex-hitman comes out of retirement to seek revenge on gangsters.",
        "Toys come to life and embark on adventures in a child's room.",
        "A hobbit's quest to destroy a powerful ring threatens Middle-earth.",
        "A lion cub learns to embrace his destiny as king of the jungle.",
        "A thief who steals corporate secrets through dream-sharing technology.",
        "A team travels through a wormhole to save humanity from extinction.",
        "A Roman general seeks vengeance after his family is murdered.",
        "A tragic love story set aboard the ill-fated Titanic ship.",
        "Humans colonize Pandora, facing a conflict with native inhabitants.",
        "Batman battles the Joker to save Gotham City.",
        "Stories of crime and redemption intersect in unexpected ways.",
        "An insomniac office worker forms an underground fight club.",
        "A man with a low IQ influences many lives with his simple wisdom.",
        "Two imprisoned men bond over years, finding redemption together.",
        "The aging patriarch of a crime family transfers control to his son.",
        "Superheroes team up to save Earth from an alien invasion.",
        "Dinosaurs are brought back to life in a theme park with disastrous results.",
        "A space opera about the battle between the Jedi and Sith.",
        "A teenager travels back in time in a DeLorean car.",
        "A young FBI cadet seeks help from a cannibalistic serial killer.",
        "Allied soldiers undertake a risky mission during World War II.",
        "A businessman saves hundreds during the Holocaust by employing Jews.",
        "A Scottish warrior leads a rebellion against English rule.",
        "A classic romantic drama set in World War II Casablanca.",
        "The rise and fall of a mob associate in New York City.",
        "An undercover cop infiltrates a crime syndicate.",
        "A jazz drummer's intense training leads to self-destruction.",
        "A musician and an actress fall in love in Los Angeles.",
        "Two magicians engage in a dangerous rivalry.",
        "The story of Facebook's founding and legal battles.",
        "A group of space criminals must save the galaxy.",
        "A wisecracking mercenary breaks the fourth wall.",
        "A girl becomes a symbol in a dystopian rebellion.",
        "Post-apocalyptic chase across the desert for survival.",
        "A superhero fights to save his nation and the world.",
        "An Amazon warrior leaves her home to fight in World War I.",
        "A young man becomes the superhero Spider-Man.",
        "An industrialist builds a powered suit to fight evil.",
        "A surgeon learns mystic arts to heal and battle threats.",
        "A family of superheroes fights to save the world.",
        "A clownfish searches for his missing son in the ocean.",
        "Monsters scare kids to generate energy for their city.",
        "An elderly man fulfills a lifelong dream with a young scout.",
        "Emotions inside a young girl's mind guide her through life.",
        "A boy journeys to the Land of the Dead to find his family.",
        "Two sisters struggle with their kingdom's magical curse.",
        "A young navigator sails to save her people and find herself.",
        "Animals live in a city and learn to coexist with humans.",
        "A rat aspires to be a chef in a Parisian restaurant."
    ]
}

df = pd.DataFrame(data)


# Initialize TF-IDF Vectorizer to convert descriptions to TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Function to get user profile vector from movies watched
def create_user_profile(watched_titles, df, tfidf_matrix):
    # Get indices of movies user watched
    indices = []
    for title in watched_titles:
        title = title.strip()
        idx_list = df.index[df['title'].str.lower() == title.lower()].tolist()
        if idx_list:
            indices.append(idx_list[0])
        else:
            print(f"Warning: '{title}' not found in the movie list.")
    if not indices:
        raise ValueError("No valid movie titles found in input.")

    # Extract TF-IDF vectors for watched movies
    watched_vectors = tfidf_matrix[indices]

    # Compute the average vector (user profile)
    user_profile = watched_vectors.mean(axis=0)

    # Convert from numpy.matrix to numpy.ndarray to avoid compatibility issues
    user_profile = user_profile.A

    return user_profile


# Function to recommend movies based on user profile
def recommend_movies(user_profile, df, tfidf_matrix, top_n=3):
    # Compute cosine similarity between user profile and all movie descriptions
    cosine_similarities = cosine_similarity(user_profile, tfidf_matrix).flatten()

    # Show similarity with all movies
    print("\nCosine similarity scores with all movies:")
    for i, score in enumerate(cosine_similarities):
        print(f"{df['title'].iloc[i]}: {score:.4f}")

    # Get indices of top recommended movies (exclude watched movies)
    top_indices = cosine_similarities.argsort()[::-1]

    return top_indices, cosine_similarities

# --- Main interactive flow ---
if __name__ == "__main__":
    watched_input = input("Enter the movie(s) you have watched (comma-separated): ")
    watched_titles = [t.strip() for t in watched_input.split(',')]

    try:
        user_profile = create_user_profile(watched_titles, df, tfidf_matrix)
        print(f"\nUser profile vector shape: {user_profile.shape}")

        top_indices, similarities = recommend_movies(user_profile, df, tfidf_matrix)

        # Filter out the movies already watched
        watched_indices = [df.index[df['title'].str.lower() == t.lower()][0] for t in watched_titles if t.lower() in df['title'].str.lower().values]

        # Recommend top 3 movies not watched yet
        recommendations = [i for i in top_indices if i not in watched_indices][:3]

        print("\nRecommended movies for you:")
        for idx in recommendations:
            print(f"{df['title'].iloc[idx]} (Similarity: {similarities[idx]:.4f})")

    except ValueError as e:
        print(e)