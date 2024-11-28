import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import joblib

data = pd.read_csv('data/dataset.csv')


user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

data['user_id'] = user_encoder.fit_transform(data['user_id'])
data['product_id'] = product_encoder.fit_transform(data['product_id'])


user_item_matrix = data.pivot(
    index='user_id', columns='product_id', values='rating').fillna(0)

similarity = cosine_similarity(user_item_matrix)
recommendation_model = {'similarity': similarity,
                        'user_encoder': user_encoder, 'product_encoder': product_encoder}


joblib.dump(recommendation_model, 'models/recommendation_model.pkl')

print("Model training complete. Saved to models/recommendation_model.pkl.")
