import torch
from flask import Flask, request, jsonify, Response

from config import CONFIG
from aifr.models.multitask_dal.model import Multitask_DAL
from utils.similarity_handler import SimilarityHandler


# Create a Flask application instance
app = Flask(__name__)

# Get the configuration instance (Singleton)
config = CONFIG.instance()

# Initialize the model with configuration parameters
model = Multitask_DAL(embedding_size=512, number_of_classes=500, margin_loss_name=config['margin_loss'])

# Load model weights if provided in the configuration
if config["model_weights"]:
    model.load_state_dict(torch.load(config["model_weights"], map_location='cpu'))
    print(f'Loaded weights from {config["model_weights"]}')

# Set the model to evaluation mode
model.eval()


@app.route('/')
def home():
    """
    Home route of the API.

    Returns:
        str: A welcome message.
    """
    return "MPAIFR REST API"


@app.route('/images_similarity', methods=['POST'])
def get_similarity():
    """
    Endpoint to calculate similarity between two images.

    Expects 'image1' and 'image2' to be provided in the request files.

    Returns:
        Response: JSON response with the similarity score or error message.
    """
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"invalid_request_error": "Please provide two images."}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if not file1 or not file2:
        return jsonify({"invalid_request_error": "Invalid files provided."}), 400

    similarity = SimilarityHandler.get_response(model, file1, file2)

    if similarity:
        return jsonify({"similarity": round(similarity, 2)})

    return jsonify({"request_error": "The parameters were valid but the request failed."}), 402


@app.route('/batch_images_similarity', methods=['POST'])
def get_batch_similarity():
    """
    Endpoint to calculate average similarity between two lists of images.

    Expects 'imageList1' and 'imageList2' to be provided in the request files.

    Returns:
        Response: JSON response with the average similarity score or error message.
    """
    if 'imageList1' not in request.files or 'imageList2' not in request.files:
        return jsonify({"invalid_request_error": "Please provide the images."}), 400

    image_list_1_files = request.files.getlist('imageList1')
    image_list_2_files = request.files.getlist('imageList2')

    if not image_list_1_files:
        return jsonify({"invalid_request_error": "Please provide a list of images."}), 400
    if not image_list_2_files:
        return jsonify({"invalid_request_error": "Please provide a second list of images."}), 400

    similarities = []

    for file1 in image_list_1_files:
        for file2 in image_list_2_files:
            if file1 and file2:
                similarity = SimilarityHandler.get_response(model, file1, file2)
                similarities.append(similarity)
            else:
                return jsonify({"invalid_request_error": "Invalid files provided."}), 400

    if similarities:
        average_similarity = sum(similarities) / len(similarities)
        return jsonify({"similarity": round(average_similarity, 2)})

    return jsonify({"request_error": "The parameters were valid but the request failed."}), 402

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='3000', debug=True, use_reloader=False, threaded=True)
