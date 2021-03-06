import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './upload/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Er is helaas iets misgegaan', 'error')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('Geen bestand geselecteerd', 'error')
            return redirect(request.url)

        # check if file is eligible for upload
        if not allowed_file(file.filename):
            flash('Dit bestand mag je helaas niet uploaden', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return render_template('upload.html')
    # return '''
    # <!doctype html>
    # <title>Upload new File</title>
    # <h1>Upload new File<h1>
    # <form method=post enctype=multipart/form-data>
    #   <input type=file name=file>
    #   <input type=submit value=Upload>
    # </form>
    # '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
