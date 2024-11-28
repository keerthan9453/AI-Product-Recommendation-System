from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('models/recommendation_model.pkl')


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')

    if user_id is None:
        return jsonify({'error': 'User ID is required'}), 400

    try:
        # Get similarity matrix and encoders
        similarity = model['similarity']
        user_encoder = model['user_encoder']
        product_encoder = model['product_encoder']

        # Encode user
        encoded_user = user_encoder.transform([user_id])[0]

        # Recommend top 5 products for the user
        user_similarities = similarity[encoded_user]
        top_products = np.argsort(user_similarities)[-5:][::-1]

        recommended_products = product_encoder.inverse_transform(top_products)
        return jsonify({'recommended_products': recommended_products.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
