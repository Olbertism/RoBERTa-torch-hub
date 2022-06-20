from flask import Flask, request
from flask_cors import CORS, cross_origin
import time
import torch

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
# -> ideally, in this init it should load if called first time, but then be kept suspended. Id like to have the model on the server in a constant "loaded" state to keep time down. We will see if this is possible...
roberta.eval()

def get_prediction(tokens_list):
        '''
        tokens_list: expects a list with list, each sublist 2 items
        '''
        predictions = []
        with torch.no_grad():

            start = time.perf_counter()

            for tokens in tokens_list:
                encoded_prompt = roberta.encode(tokens[0], tokens[1])
                prediction = roberta.predict('mnli', encoded_prompt).argmax().item()

                predictions.append(prediction)

            end = time.perf_counter()
            print(f"Predictions performed in {end - start:0.4f} seconds")

            return predictions



@app.route("/")
@cross_origin()
def home():
    return "You are in the root directory"


@app.route("/predict", methods = ['POST'])
@cross_origin()
def prediction():
  if request.method == 'POST':
    request_data = request.get_json()
    # accept a json body
    # parse json
    # request_data = { "prompts" : [['Mars is a planet', 'Mars is a planet'], ['Mars is a planet', 'Mars is # not a planet']]}

    print(request_data["prompts"])

    # call the model function, pass in prompts from body

    predictions = get_prediction(request_data["prompts"])
    print(predictions)
    # return the predictions, incl. status code

    return {"predictions": predictions}, 200
  else:
    return "Method not allowed", 405
