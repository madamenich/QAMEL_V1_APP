from flask import Flask, request, jsonify, Response
import json
import subprocess
from utils import conversion as utils

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/extract', methods=['POST'])
def extract_nodes():
    return utils.extract_json(request)


@app.route('/qml', methods=['POST'])
def run_qml():
    # request_body = request.json
    request_body = utils.extract_json(request).json
    print(request_body.get("datasets_type", 'MNIST'))

    dataset = request_body.get("datasets_type", 'MNIST')
    batch_size = request_body.get("batch_size", 1)
    n_samples = request_body.get("number_of_samples", 100)
    seed = request_body.get("random_seed", None)  # Get the value associated with "random_seed"
    if seed is None or seed == 0 or seed == "":
        seed = 42
    fractions = request_body.get("split_ratio", 0.7)
    classes = request_body.get("labels", [0, 1])
    device = request_body.get("is_cuda", 'cpu')
    epochs = request_body.get("train_epochs", 10)
    lr = request_body.get("lr", 0.001)
    command = " ".join([
        "python3",
        "qmnist.py",
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        "--n_samples", str(n_samples),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--dataset", str(dataset),
        "--fraction", str(fractions),
        "--classes", ",".join(map(str, classes)),  # Assuming classes is a list
        "--cuda", str(device)
    ])
    print('command',command)
    # completed_process = subprocess.run(["python3", "qmnist.py",
    #                                     "--batch_size", str(batch_size),
    #                                     "--seed", str(seed), "--n_samples",
    #                                     str(n_samples), "--epochs", str(epochs),
    #                                     "--lr", str(lr),
    #                                     "--dataset", str(dataset),
    #                                     "--fraction", str(fractions),
    #                                     "--classes", str(classes),
    #                                     "--cuda", str(device)
    #                                     ],
    #                                    capture_output=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    def generate():
        while True:
            line = process.stdout.readline()
            if line:
                yield line.decode('utf-8')
            else:
                break

    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
