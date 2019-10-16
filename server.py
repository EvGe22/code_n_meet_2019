from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def main_page():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['upload_image']
    print('ha')
    return render_template('result_page.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
