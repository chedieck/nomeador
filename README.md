# Nomeador

Simple name generator with LSTM's.
-
```
usage: nomeador.py [-h] [-gn GENERATE_NAMES] [-go GENERATE_ORIGINALS] [-ga]
                   [-t TRAIN_MODEL] [-l LOOKBACK] [-m MODEL]

Simple name generator. Uses LSTM's, which are not ideal for data generation

optional arguments:
  -h, --help            show this help message and exit
  -gn GENERATE_NAMES, --generate-names GENERATE_NAMES
                        generates N random names
  -go GENERATE_ORIGINALS, --generate-originals GENERATE_ORIGINALS
                        generates N original random names, can take a while to run
                        depending on the model
  -ga, --generate-all   generates a name or each letter of the alphabet. Depends
                        exclusively on the model
  -t TRAIN_MODEL, --train-model TRAIN_MODEL
                        trains a new <model>.h5, requires --lookback
  -l LOOKBACK, --lookback LOOKBACK
                        lookback to be used on model training or prediction, default
                        is 7
  -m MODEL, --model MODEL
                        <model>.h5 to be used on prediciton, defaults is default
```

This repository is intended to be replicated in the future, using GAN's instead of LSTM's.

Examples:

`python -ga`

`python -go 3`

`python -t teste -l 10`

`python -ga -m teste -l 10`

-

Remember, if you train a new model `-t example` using some `--lookback == L != 7`, it is necessary to specify `-l L` when using `-m example`.

