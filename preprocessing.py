import pandas as pd
from random import choice
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_names(sex, freq=0.03):
    dataset = pd.read_csv('grupos.csv')
    todas_mulheres = dataset[dataset.classification == 'F'].sort_values('frequency_female', ascending=0)
    todos_homens = dataset[dataset.classification == 'M'].sort_values('frequency_male', ascending=0)

    #mais frequentes
    border_m  = int(freq * len(todos_homens))
    border_f  = int(freq * len(todas_mulheres))
    homens =  todos_homens.iloc[0:border_m, :].name 
    mulheres =  todas_mulheres.iloc[0:border_f, :].name

    if sex.lower() == 'm':
        names = homens
    elif sex.lower() == 'f':
        names = mulheres
    else:
        names = pd.concat([homens, mulheres], ignore_index=True)

    names = names.apply(lambda x: x + '0')
    
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(names)
    sequences = tokenizer.texts_to_sequences(names)
    padded = pad_sequences(sequences)

    return {'tokenizer': tokenizer, 'padded': padded, 'names': names}







