from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
import time
import os
import torch
from dotenv import load_dotenv
from fairseq.models.roberta import RobertaModel

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

load_dotenv()

# Size around 750MB
roberta = RobertaModel.from_pretrained('model/roberta.large.mnli', checkpoint_file='model.pt')
roberta.eval()

def get_prediction(tokens_list):
        '''
        tokens_list: expects a list with dicts
        [{ "source": "string", "comparer": "string }, ...]
        '''
        predictions = []
        with torch.no_grad():

            start = time.perf_counter()

            for tokens in tokens_list:
                encoded_prompt = roberta.encode(tokens["source"], tokens["comparer"])
                prediction = roberta.predict('mnli', encoded_prompt).argmax().item()

                predictions.append(prediction)

            end = time.perf_counter()
            print(f"Predictions performed in {end - start:0.4f} seconds")

            return predictions


@app.route("/")
@cross_origin()
def home():
    return "You are in the root directory"

@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

###
# KEYS: 0: CONTRADICTION  1: NEUTRAL  2: ENTAILMENT
##

@app.route("/predict", methods = ['POST'])
@cross_origin()
def predict():
  if request.method == 'POST':

    """ args = request.args.to_dict()
    print(args)
    test = os.environ.get('ROBERTAKEY')
    print("------------------------------")
    print(test)
    if args.get("api-key") != os.environ.get('ROBERTAKEY'):
      return {"status": "error", "message": "Request forbidden"}, 403 """

    request_data = request.get_json()

    # old shape
    # request_data = { "prompts" : [['Mars is a planet', 'Mars is a planet'], ['Mars is a planet', 'Mars is # not a planet']]}

    # new shape
    # {"instances": [{ "image": X.tolist() }]}

    print(request_data["instances"])

    # call the model function, pass in prompts from body

    try:
      predictions = get_prediction(request_data["instances"])
      print(predictions)
      return {"status": "ok", "message" : "Request successful", "predictions": predictions}, 200

    except Exception:
      return {"status": "error", "message": "Predictions failed"}, 500
  else:
    return {"status": "error", "message": "Method not allowed"}, 405


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)

"""
From Github Issue https://github.com/facebookresearch/fairseq/pull/4440

install_requires=[
            "cffi",
            "cython",
            'dataclasses; python_version<"3.7"',
            "hydra-core>=1.0.7,<1.1",
            "omegaconf<2.1",
            'numpy<1.20.0; python_version<"3.7"',
            'numpy; python_version>="3.7"',
            "regex",
            "sacrebleu>=1.4.12",
            "torch",
            "tqdm",
            "bitarray",
            "torchaudio>=0.8.0",
        ],

Not all of those are needed. Most important are the correct versions of hydra-core (install this first) and omegaconf"""


"""
Model source:
https://github.com/facebookresearch/fairseq
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}

"""