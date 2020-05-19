from src.finetune_he_sentiment import get_args, main
from src import raw_to_pickle

DATASET_PATH = '/root/heMoji/data/amram_2017/'
LOGS_DIR = '/root/heMoji/data/amram_2017/'
PRETRAINED_PATH = '/root/heMoji/model/model.h5'
VOCAB_PATH = '/root/heMoji/model/vocab.json'
EPOCHS = 100
TRANSFER = 'chain-thaw'


if __name__ == '__main__':
    """Finetuning example.

    Trains the heMoji model on the he sentiment tweeter dataset,
    using the 'last', 'chain-thaw' and 'add-last' finetuning method and the accuracy metric.

    The 'last' method (transfer param) does the following:
    0) Load all weights except for the softmax layer. Do not add tokens to the
       vocabulary and do not extend the embedding layer.
    1) Freeze all layers except for the softmax layer.
    2) Train.

    The 'chain-thaw' method (transfer param) does the following:
    0) Load all weights except for the softmax layer. Extend the embedding layer if
       necessary, initialising the new weights with random values.
    1) Freeze every layer except the last (softmax) layer and train it.
    2) Freeze every layer except the first layer and train it.
    3) Freeze every layer except the second etc., until the second last layer.
    4) Unfreeze all layers and train entire model.

    """
    params = get_args(DATASET_PATH, LOGS_DIR, PRETRAINED_PATH, VOCAB_PATH, EPOCHS, TRANSFER)
    raw_to_pickle.process(params['data_path'])
    main(params)
