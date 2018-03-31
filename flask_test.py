import os
from flask import render_template
from flask import Flask, request, redirect, url_for
from werkzeug.utils  import secure_filename

UPLOAD_FOLDER = '/tmp/upload'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return render_template('index.html')

@app.route('/ajax_process', methods=['POST'])
def ajax_process():
    file = request.files['upload_image']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return "文件（%s）是狗，概率：%.4f"%(filename, 0.39399393)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')