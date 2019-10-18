from tempfile import mktemp

import cv2
import base64

from icons import find_icons, ICONS
from match import draw_match, resize
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def main_page():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['upload_image']
    tmp_file_path = mktemp(prefix='saved_file', suffix='.jpg')
    f.save(tmp_file_path)

    matches = find_icons(tmp_file_path)
    image = cv2.imread(tmp_file_path)
    image, _ = resize(image, height=600)
    for icon_name, match in matches.items():
        draw_match(image, match.detection_bbox, icon_name)

    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer)
    matches_to_display = {k: ICONS[k].description for k in matches.keys()}
    return render_template('result_page.html', image_base64=image_base64.decode('utf-8'),
                           matches=matches_to_display)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
