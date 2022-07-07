import time
import os
import torch


class PythonPredictor:

    def __init__(self):
        """ Download pretrained model. """
        self.model = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
        self.model.eval()

    def predict(self, tokens_list):
        '''
        tokens_list: expects a list with list, each sublist 2 items
        '''
        predictions = []
        with torch.no_grad():

            start = time.perf_counter()

            for tokens in tokens_list:
                encoded_prompt = self.model.encode(tokens[0], tokens[1])
                prediction = self.model.predict('mnli', encoded_prompt).argmax().item()

                predictions.append(prediction)

            end = time.perf_counter()
            print(f"Predictions performed in {end - start:0.4f} seconds")

            return predictions