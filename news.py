import json
import argparse
import time
import numpy as np
import re
import pandas as pd
from collections import Counter


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

    print("Starting:", time.ctime())

    ############################################
    # Preprocessing of data

    # json_content = open('News_Category_Dataset_v2.json', "r").read()
    # list_of_dicts = [json.loads(str(item)) for item in json_content.strip().split('\n')]

    news = pd.read_json('News_Category_Dataset_v2.json', lines=True)

    # clean the data
    news['clean_hd'] = news['headline'].apply(lambda x: headline_to_words(x))

    cat_set = set(news['category'])
    cat_list = list(cat_set)
    cat_to_int = {cat: ii for ii, cat in enumerate(cat_list, 1)}

    # labels
    cat_array = []
    for cat in news['category']:
        cat_array.append(cat_to_int[cat])
    labels = np.array(cat_array)

    # headlines
    all_text = ' '.join(news['clean_hd'])
    words = all_text.split()

    # Convert words to integers
    counts = Counter(words)

    numwords = 200  # Limit the number of words to use
    vocab = sorted(counts, key=counts.get, reverse=True)[:numwords]
    # print(vocab)
    # print(vocab)
    # print('\n\n')
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    # print(vocab_to_int)

    hd_ints = []
    for each in news['clean_hd']:
        hd_ints.append([vocab_to_int[word] for word in each.split() if word in vocab_to_int])
    # print(hd_ints[:20])

    # print(len(labels))
    # print(len(hd_ints))

    hd_len = Counter([len(x) for x in hd_ints])
    print("Zero-length reviews: {} of {} ({}%)".format(hd_len[0], len(hd_ints), round(hd_len[0] * 100.0 / len(hd_ints), 2)))
    print("Maximum tweet length: {}".format(max(hd_len)))

    # Remove those tweets with zero length and its corresponding label
    hd_idx = [idx for idx, hd in enumerate(hd_ints) if len(hd) > 0]
    labels = labels[hd_idx]
    news = news.iloc[hd_idx]
    hd_ints = [hd for hd in hd_ints if len(hd) > 0]

    # print(hd_ints[:30])

    seq_len = max(hd_len)
    features = np.zeros((len(hd_ints), seq_len), dtype=int)
    for i, row in enumerate(hd_ints):
        features[i, -len(row):] = np.array(row)[:seq_len]
    # print(features[:30])

    split_frac = 0.8
    split_idx = int(len(features) * 0.8)
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

    # print(len(labels))
    # print(len(hd_ints))

    # labels
    # news = {'category': [], 'headline': []}
    # for d in list_of_dicts:
    #     news['category'].append(d['category'])
    #     news['headline'].append(d['headline'])
    #
    # cat_set = set(news['category'])
    # cat_list = list(cat_set)
    # cat_to_int = {cat: ii for ii, cat in enumerate(cat_list, 1)}
    #
    # news['cat_int'] = []
    # for cat in news['category']:
    #     news['cat_int'].append(cat_to_int[cat])
    #
    # # Create a list of labels
    # labels = np.array(news['cat_int'])
    #
    # # headlines
    #
    # news['clean_hdline'] = []
    # for hdline in news['headline']:
    #     news['clean_hdline'].append(headline_to_words(hdline))
    # # print(news['headline'][:20])
    # # print(news['clean_hdline'][:20])
    #
    # all_text = ' '.join(news['clean_hdline'])
    # words = all_text.split()
    #
    # # Convert words to integers
    # counts = Counter(words)
    #
    # numwords = 200  # Limit the number of words to use
    # # print(sorted(counts, key=counts.get, reverse=True))
    # vocab = sorted(counts, key=counts.get, reverse=True)[:numwords]
    # # print(vocab)
    # # print(vocab)
    # vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    # # print(vocab_to_int)
    #
    # hdline_ints = []
    # for each in news['clean_hdline']:
    #     hdline_ints.append([vocab_to_int[word] for word in each.split() if word in vocab_to_int])
    # # print(hdline_ints[:20])
    #
    # hdline_len = Counter([len(x) for x in hdline_ints])
    # print("Zero-length reviews: {} of {} ({}%)".format(hdline_len[0], len(hdline_ints), round(hdline_len[0] * 100.0 / len(hdline_ints), 2)))
    # print("Maximum tweet length: {}".format(max(hdline_len)))
    #
    # # Remove those tweets with zero length and its corresponding label
    # hdline_idx = [idx for idx, hdline in enumerate(hdline_ints) if len(hdline) > 0]
    # labels = labels[hdline_idx]
    # Tweet = Tweet.iloc[tweet_idx]
    # tweet_ints = [tweet for tweet in tweet_ints if len(tweet) > 0]

    ################################################################################
    # print(all_text[:100])
    # print('\n\n\n\n')
    # print(words)

    # hdlines_ints = []
    # for each in news['headline']:
    #     hdlines.append([vocab_to_int[word] for word in each.split() if word in vocab_to_int])
    # print(tweet_ints)

        # news['category'].apply(lambda x: cat_to_int[x])
    # print(news['category'])
    # print(news['cat_int'])
    # Tweet['sentiment'] = Tweet['twsentiment'].apply(lambda x: 0 if x == 'negative' else 1 if x == 'positive' else 2)
    # print(cat_to_int)
    # for i in news['category']:
    #     print(cat_to_int[i])
    #print(news['category'])
    #print(news['headline'])
