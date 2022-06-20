import torch
import time


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
# Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation """

def predict(tokens_list):
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


""" with torch.no_grad():

    start = time.perf_counter()
    tokens_list = [['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
                   ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
                   ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.']]

    for tokens in tokens_list:
        encoded_prompt = roberta.encode(tokens[0], tokens[1])
        prediction = roberta.predict('mnli', encoded_prompt).argmax().item()
        print(prediction)

    end = time.perf_counter()
    time_result = (end - start)
    print(f"Loop statements performed in {end - start:0.4f} seconds") """




class ModelInitFailedError(Exception):
    pass


# ok, I need to check on the thread class code how thats handled there... way simpler if I rememeber correctly
class Roberta(object):

    __instance = None

    def __new__(cls, *args, **kargs):
        return cls.getInstance(cls, *args, **kargs)

    def __init__(self):
        pass

    def getInstance(cls, *args, **kargs):
        '''Static method to have a reference to **THE UNIQUE** instance'''
        # Critical section start
        # cls.__lockObj.acquire()
        try:
            if cls.__instance is None:
                # (Some exception may be thrown...)
                # Initialize **the unique** instance
                cls.__instance = object.__new__(cls, *args, **kargs)

                '''Initialize object **here**, as you would do in __init__()...'''

        except Exception:
            raise ModelInitFailedError
        # finally:
            #  Exit from critical section whatever happens
            # cls.__lockObj.release()
        # Critical section end

        return cls.__instance

    def predict(self, tokens_list):
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


    getInstance = classmethod(getInstance)
