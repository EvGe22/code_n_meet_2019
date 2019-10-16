from flask import Flask
from flask import render_template
from flask import request

app = Flask(__name__, )


@app.route('/')
def main_page():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['the_file']

    return render_template('result_page.html')


app.run(port=5001)
