import nltk
import string
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Preloaded English stopwords set to avoid repeated initialization
EN_STOPWORDS = set(stopwords.words('english'))

# Fixed Word2Vec model parameters
VECTOR_SIZE = 100
WINDOW_SIZE = 5
MIN_WORD_COUNT = 1
USE_SKIP_GRAM = 1
NEGATIVE_SAMPLING = 5
TRAIN_EPOCHS = 200

raw_text = """
Daily Life, Nature and Travel
In the hustle and bustle of modern urban life, many people forget to pause and appreciate the beauty of nature that surrounds them—yet travel offers a perfect chance to reconnect with the natural world beyond the city limits. Every morning, as the sun rises over the city skyline, golden light filters through the leaves of old oak trees in the neighborhood park, but for those who love to travel, the sunrise over a mountain lake or a coastal beach is an even more precious sight. Families walk their dogs along winding paths in local parks, but on weekends and holidays, they often travel to nearby towns, national parks, or rural villages to escape the noise of the city and breathe fresh air.

Travel is not just about visiting new places; it is about exploration, adventure, and learning. A short trip to a countryside village can teach you about local traditions, while a longer journey to a foreign country opens your eyes to different cultures, languages, and ways of life. When you travel through mountain ranges, you hike along trails lined with pine trees and wild berries, listen to the sound of mountain streams, and watch eagles soar above snow-capped peaks. Travel to coastal regions lets you walk along sandy beaches, collect seashells, and taste fresh seafood caught by local fishermen. 

Many people find joy in planning their travel: researching destinations, booking accommodation, packing a backpack with essentials, and creating a list of places to visit—from historic castles and museums to hidden waterfalls and quiet forests. Even a simple day trip to a nearby lake can feel like an adventure, as you row a boat across calm water, fish for trout, or have a picnic with family and friends. Travel also teaches patience: delayed trains, unexpected weather, or language barriers are small challenges that make the journey more memorable.

Nature is the greatest companion for travel. When you travel to a national park, you encounter deer grazing in meadows, hear the call of woodpeckers in forests, and smell the sweet scent of pine and cedar in the air. In spring, travel to cherry blossom groves in Japan or tulip fields in the Netherlands, and in autumn, travel to New England to see maple leaves turn fiery red and orange, crunching underfoot as you walk through quiet woods. 

Reading is another way to fuel your desire to travel—books about travel memoirs, adventure novels, or guidebooks transport you to far-off lands, from the streets of Paris to the mountains of Nepal, even when you cannot leave your home. A good travel book, like a real trip, can make time slow down, allowing your mind to wander and dream of future journeys. 

Food is an essential part of travel too. When you travel, you taste local dishes: fresh pasta in Italy, spicy tacos in Mexico, or steaming bowls of ramen in Japan. Farmers’ markets in the cities you travel to offer juicy strawberries, crisp lettuce, and ripe tomatoes, while street vendors sell warm bread, savory pastries, and sweet treats that reflect the local culture. Sharing a meal with strangers you meet while traveling creates connections that last a lifetime.

As the day ends and the sun sets, whether you are at home or traveling in a foreign land, the sky painted in hues of purple and orange brings a sense of peace. Travel reminds us that happiness is found not just in routine daily life, but in the small moments of adventure: a bird’s song in a foreign forest, a breeze through an open window of a train as you travel across the countryside, or the taste of a fresh mango from a market in Thailand. Taking the time to travel, explore, and connect with nature makes even the most ordinary days feel rich and meaningful.
"""

#Modular Text Preprocessing Functions
def convert_lower(text: str) -> str:
    """Convert input text to all lowercase characters"""
    return text.lower()

def remove_punctuation(text: str) -> str:
    """Remove all punctuation symbols from the text"""
    punctuation_map = str.maketrans('', '', string.punctuation)
    return text.translate(punctuation_map)

def tokenize_words(text: str) -> list:
    """Split text into individual word tokens"""
    return word_tokenize(text)

def filter_useless_tokens(tokens: list) -> list:
    """Filter out stopwords and empty whitespace tokens"""
    return [token.strip() for token in tokens if token.strip() and token not in EN_STOPWORDS]

def full_preprocess(text: str) -> list:
    """
    Complete text preprocessing pipeline
    Output format is EXACTLY the same as the original code
    """
    # Follow the exact preprocessing sequence of the original code
    step1 = convert_lower(text)
    step2 = remove_punctuation(step1)
    step3 = tokenize_words(step2)
    clean_tokens = filter_useless_tokens(step3)
    # Critical: Combine all tokens into a single sentence
    corpus = [clean_tokens]
    return corpus

#Main Execution Flow
if __name__ == "__main__":
    # 1. Run full text preprocessing
    corpus_data = full_preprocess(raw_text)
    clean_word_list = corpus_data[0]

    # 2. Display preprocessing results
    print(f"Total number of words after preprocessing：{len(clean_word_list)}")
    print(f"Preprocess the first 20 words：{clean_word_list[:20]}")

    # 3. Train Word2Vec model with fixed parameters
    word2vec_model = Word2Vec(
        sentences=corpus_data,
        vector_size=VECTOR_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_WORD_COUNT,
        sg=USE_SKIP_GRAM,
        negative=NEGATIVE_SAMPLING,
        epochs=TRAIN_EPOCHS
    )

    # 4. Query top 10 most similar words to 'travel'
    print("\n10 words most semantically similar to travel：")
    similar_words = word2vec_model.wv.most_similar("travel", topn=10)
    for idx, (word, score) in enumerate(similar_words, 1):
        print(f"{idx:2d}. {word:<15} Similarity：{score:.4f}")

    # 5. Display the first 10 dimensions of the 'travel' word vector
    travel_vector = word2vec_model.wv["travel"][:10]
    print("travel Word vectors (first 10 dimensions)：")
    print([round(num, 4) for num in travel_vector])