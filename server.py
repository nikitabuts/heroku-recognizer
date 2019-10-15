from flask import Flask, render_template
import os

app = Flask(__name__)



@app.route("/", methods=["POST", "GET", 'OPTIONS'])
def index_page():
    return render_template('index.html')



port = int(os.environ.get("PORT", 4000))
app.debug = True
app.run(host='192.168.0.11', port=port)


