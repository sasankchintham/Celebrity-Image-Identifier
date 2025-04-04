from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import logging
from util import classify_image

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

logging.basicConfig(level=logging.DEBUG)

@app.route('/classify_image', methods=['POST'])
def classify_api():
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({'error': 'Missing image data'}), 400

        try:
            image_bytes = base64.b64decode(data['image_data'])
            image_stream = io.BytesIO(image_bytes)
            image_data = image_stream.read()
        except Exception as decode_error:
            logging.error(f" Base64 Decode Error: {decode_error}")
            return jsonify({'error': 'Invalid image format'}), 400

        result = classify_image(image_data)

        if "error" in result:
            logging.warning(f" Classification issue: {result['error']}")
            return jsonify(result), 400

        logging.info(f" Prediction successful: {result}")
        return jsonify(result), 200

    except Exception as e:
        logging.error(f" Server error: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == "__main__":
    app.run(debug=True)
