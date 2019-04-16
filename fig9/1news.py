import argparse
import time
import numpy as np
import re
import pandas as pd
from utils import HistoryCheckpoint
from keras import callbacks, regularizers
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, GRU, SimpleRNN
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight


def headline_to_words(headline):
    """
    Only keeps ascii characters in the headline and discards @words

    :param headline:
    :return:
    """
    letters_only = re.sub("[^a-zA-Z@]", " ", headline)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not re.match("^[@]", w)]
    return " ".join(meaningful_words)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2

    print("Starting:", time.ctime())

    ############################################
    # Preprocessing of data

    news = pd.read_json('News_Category_Dataset_v2.json', lines=True)

    # Clean the data
    news['clean_hd'] = news['headline'].apply(lambda x: headline_to_words(x))

    cat_set = set(news['category'])
    cat_list = list(cat_set)
    n_categories = len(cat_list)
    cat_to_int = {cat: ii for ii, cat in enumerate(cat_list, 0)}

    # Labels
    cat_array = []
    for cat in news['category']:
        cat_array.append(cat_to_int[cat])
    labels = np.array(cat_array)

    # Headlines
    all_text = ' '.join(news['clean_hd'])
    words = all_text.split()

    # Convert words to integers
    counts = Counter(words)

    # numwords = 200  # Limit the number of words to use
    frac = 0.1 # in percents
    numwords = int(len(counts) * 0.01 * frac)
    vocab = sorted(counts, key=counts.get, reverse=True)[:numwords]
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    hd_ints = []
    for each in news['clean_hd']:
        hd_ints.append([vocab_to_int[word] for word in each.split() if word in vocab_to_int])

    hd_len = Counter([len(x) for x in hd_ints])
    print("Zero-length reviews: {} of {} ({}%)".format(hd_len[0], len(hd_ints), round(hd_len[0] * 100.0 / len(hd_ints), 2)))
    print("Maximum tweet length: {}".format(max(hd_len)))

    # Remove those tweets with zero length and its corresponding label
    hd_idx = [idx for idx, hd in enumerate(hd_ints) if len(hd) > 0]
    labels = labels[hd_idx]
    news = news.iloc[hd_idx]
    hd_ints = [hd for hd in hd_ints if len(hd) > 0]

    seq_len = max(hd_len)
    features = np.zeros((len(hd_ints), seq_len), dtype=int)
    for i, row in enumerate(hd_ints):
        features[i, -len(row):] = np.array(row)[:seq_len]

    # Randomize order of features and labels
    np.random.seed(seed=7)
    order = np.arange(len(features))
    np.random.shuffle(order)
    features = features[order]
    labels = labels[order]

    split_frac = 0.8
    split_idx = int(len(features) * split_frac)
    train_x, val_x = features[:split_idx], features[split_idx:]
    train_y, val_y = labels[:split_idx], labels[split_idx:]

    test_idx = int(len(val_x) * 0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    print("Train set: \t\t{}".format(train_y.shape),
          "\nValidation set: \t{}".format(val_y.shape),
          "\nTest set: \t\t{}".format(test_y.shape))

    # Weights
    class_weights = class_weight.compute_class_weight('balanced',
                                                np.unique(train_y),
                                                train_y)

    ############################################
    # Model
    drop = 0.5
    nlayers = 2  # >= 1
    RNN = LSTM  # LSTM, GRU

    neurons = 512
    embedding = 200

    model = Sequential()
    model.add(Embedding(numwords + 1, embedding, input_length=seq_len))

    if nlayers == 1:
        model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop, kernel_initializer='glorot_uniform'))
    else:
        model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop, kernel_initializer='glorot_uniform', return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop, kernel_initializer='glorot_uniform', return_sequences=True))
        model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop, kernel_initializer='glorot_uniform'))

    model.add(Dense(n_categories))
    model.add(Activation('softmax'))

    ############################################
    # Training

    learning_rate = 0.001
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    epochs = 25  # 50
    batch_size = 100

    train_y_c = np_utils.to_categorical(train_y, n_categories)
    val_y_c = np_utils.to_categorical(val_y, n_categories)

    model_path = '1models/model-{epoch:02d}-{val_loss:.2f}.hdf5'
    history_path = '1history/model-{epoch}.json'

    dump_period = 100
    if dump_period > epochs:
        dump_period = epochs

    save_model_callback = callbacks.ModelCheckpoint(model_path, period=dump_period)
    save_history_callback = HistoryCheckpoint(history_path, period=dump_period)

    model.fit(train_x, train_y_c,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(val_x, val_y_c),
              callbacks=[save_model_callback, save_history_callback],
              verbose=verbose,
              class_weight=class_weights)

    ############################################
    # Results

    test_y_c = np_utils.to_categorical(test_y, n_categories)
    score, acc = model.evaluate(test_x, test_y_c,
                                batch_size=batch_size,
                                verbose=verbose)
    print()
    print('Test ACC=', acc)

    test_pred = model.predict_classes(test_x, verbose=verbose)

    print()
    print('Confusion Matrix')
    print('-' * 20)
    print(confusion_matrix(test_y, test_pred))
    print()
    print('Classification Report')
    print('-' * 40)
    print(classification_report(test_y, test_pred))
    print()
    print("Ending:", time.ctime())
