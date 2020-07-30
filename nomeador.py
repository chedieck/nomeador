import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import Callback
import numpy as np
import preprocessing
from string import ascii_lowercase
import argparse

class myCallback(Callback):
    def __init__(self, modelname='default'):
        self.modelname = modelname
    """ salvar o modelo no meio do treino
    """
    def on_epoch_end(self, epoch, logs={}):
        self.model.save(f'modelos/{self.modelname}.h5')

def generate_windows(sequences, lookback, total_chars=27):
    x = []
    y = []
    for sentence in sequences:
        for p in range(len(sentence) - lookback):
            x.append(sentence[p:p + lookback])
            y.append(sentence[p + lookback])
    ys = to_categorical(y,num_classes=total_chars+1)
    

    return np.array(x), ys
        
def train(x,y, nunits=10, epochs=200, total_chars=27,modelname='default'):
    lookback = x.shape[1]
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = Sequential()
    model.add(Embedding(total_chars, 16, input_length=lookback))
    model.add(Bidirectional(LSTM(nunits)))
    model.add(Dense(total_chars, activation='softmax'))
    adam = Adam(lr=0.005)

    model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])

    print(model.summary())
    history = model.fit(x,y, epochs=epochs, callbacks=[myCallback(modelname)]) 
    return history, model

def frase_encode(string):
    """
    transforma a string em uma lista de tokens
    """
    return [index[s] for s in string.split()]

def frase_decode(sequence):
    """ transforma uma lista de tokens em uma string
    """
    return ' '.join([invindex[s] for s in sequence])


def prever_proximas(startstring, model):
    aux = frase_encode(startstring)
    previsões =[]
    for _ in range(100):
        pred_token = np.argmax(model.predict(np.array([aux])))
        pred_word = frase_decode([pred_token]) #próxima letra
        if pred_word ==  '0':
            break
        previsões.append(pred_word)
        
        aux.append(pred_token)
        aux = aux[1:]

    return previsões

def generate_cada_letra(model, LB):
    for char in ascii_lowercase:
        start = ' '.join(['*'] * (LB -1)) + ' ' + char
        name = char + ' ' + ' '.join(prever_proximas(start, model))
        print (name.replace(' ', '').capitalize())


def generate_name(x, model, verbose=1):
    """ Gera um nome a partir de uma seed existente aleatória.
    """
    random_start = ' '.join(['*'] * x.shape[1]) #caso que é tudo padding
    while random_start == ' '.join(['*'] * x.shape[1]): 
        random_start = frase_decode(x[np.random.choice(len(x))])

    start = random_start.replace('*', '') 
    name = start + ' ' + ' '.join(prever_proximas(random_start, model))
    if verbose:
        print(f'(seeded with {start:^20})')
    return (name.replace(' ', '').capitalize())


def generate_original(x, model, names):
    """ Gera um nome necessariamente original.
    Note que a função demorará para rodar se o modelo
    utilizado estiver muito overfitted.
    """
    while 1:
        aux = generate_name(x, model, verbose=0)
        if aux.lower() not in set(names.apply(lambda x: x[:-1].lower())):
            break
    return aux

def init_argparse():
    parser = argparse.ArgumentParser(description="Simple name generator. Uses LSTM's, which are not ideal for data generation")
    parser.add_argument('-gn', '--generate-names', type=int, help='generates N random names')
    parser.add_argument('-go', '--generate-originals', type=int, help='generates N original random names, can take a while to run depending on the model')
    parser.add_argument('-ga', '--generate-all', action='store_true', help='generates a name or each letter of the alphabet. Depends exclusively on the model')

    parser.add_argument('-t', '--train-model', help='trains a new <model>.h5, requires --lookback')
    parser.add_argument('-l', '--lookback', type=int, help='lookback to be used on model training or prediction, default is 7')
    parser.add_argument('-m', '--model', help='<model>.h5 to be used on prediciton, defaults is default')

    args = parser.parse_args()
    if args.train_model and not args.lookback:
        parser.error("--train-model requires --lockback.")

    return args

def main(args):
    if not args.train_model:
        #LB default to 7
        LB = 7 if not args.lookback else args.lookback
        model_name = 'default' if not args.model else args.model

        x, y = generate_windows(padded, LB, total_chars=total_chars)
        m = load_model(f'modelos/{model_name}.h5')
        if args.generate_names:
            print(f"Generating {args.generate_names} random names...\n")
            for _ in range(args.generate_names):
                print(generate_name(x, m))
        if args.generate_originals:
            print(f"Generating {args.generate_originals} original random names...\n")
            for _ in range(args.generate_originals):
                print(generate_original(x, m, names))
        if args.generate_all:
            print(f"Generating a name for each letter...\n")
            generate_cada_letra(m, LB)

    
    else:
        LB = args.lookback
        modelname = args.train_model
        x, y = generate_windows(padded, LB, total_chars=total_chars)
        h, m = train(x, y, total_chars=total_chars+1, epochs=400,
            modelname=modelname)


if __name__ == '__main__':
    #cria os índices
    tokenizer, padded, names = preprocessing.get_names('m').values()
    index = tokenizer.word_index
    invindex = {v:k for k, v in tokenizer.word_index.items()}
    invindex[0] = '*'
    index['*'] = 0
    total_chars = len(index) -  1

    main(init_argparse())
    



