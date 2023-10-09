from flask import Flask, request
import json
import subprocess
import utils as utils
app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/extract', methods=['POST'])
def extract_nodes():
    return utils.extract_json(request)


@app.route('/qml', methods=['POST'])
def run_qml():
    request_body = request.json
    batch_size = request_body.get("batch_size")
    train_steps = request_body.get("train_steps")
    test_steps = request_body.get("test_steps")
    epochs = request_body.get("epochs")
    lr = request_body.get("lr")
    completed_process = subprocess.run(["python3", "qmnist.py", "--batch_size", str(batch_size),
                                        "--train", str(train_steps), "--test", str(test_steps), "--epochs", str(epochs),
                                        "--lr", str(lr)], capture_output=True)

    if completed_process.returncode != 0:
        return "Fail to train the model"
    else:
        return "Success to train the model"


if __name__ == '__main__':
    app.run(debug=True)
