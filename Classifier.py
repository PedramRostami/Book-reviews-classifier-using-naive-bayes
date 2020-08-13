import pandas as pd
import string

reviews_train_path = 'dataset/reviews_train.csv'
test_data_path = 'dataset/reviews_test.csv'
stop_words_path = 'dataset/sw.txt'


def pre_processing():
    data = pd.read_csv(reviews_train_path)
    stop_words = []
    word_counts = {}
    pos_labels = data[data.Label == 'pos'].shape[0]
    neg_labels = data[data.Label == 'neg'].shape[0]
    words_in_pos_labels = 0
    words_in_neg_labels = 0
    with open(stop_words_path, 'r') as file:
        for line in file:
            for word in line.split():
                stop_words.append(word)
    for index, row in data.iterrows():
        s = str(row['Review'])
        replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        s = s.translate(replace_punctuation)
        s = s.lower()
        for word in s.split():
            if word not in stop_words:
                if word in word_counts:
                    word_counts[word] = word_counts[word] + 1
                else:
                    word_counts[word] = 1
    bag_of_words = {}
    for word in word_counts.keys():
        bag_of_words[word] = {'pos': 0, 'neg': 0}
    for index, row in data.iterrows():
        s = str(row['Review'])
        replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        s = s.translate(replace_punctuation)
        s = s.lower()
        words_line_set = set()
        for word in s.split():
            if word not in stop_words:
                words_line_set.add(word)
        words_line_list = list(words_line_set)
        if row['Label'] == 'pos':
            for word in words_line_list:
                if bag_of_words[word]['pos'] == 0:
                    words_in_pos_labels += 1
                bag_of_words[word]['pos'] += 1
        elif row['Label'] == 'neg':
            for word in words_line_list:
                if bag_of_words[word]['neg'] == 0:
                    words_in_neg_labels += 1
                bag_of_words[word]['neg'] += 1

    return bag_of_words, pos_labels, neg_labels, words_in_pos_labels, words_in_neg_labels


def classify(bag_of_words, pos_prob, neg_prob):
    data = pd.read_csv(test_data_path)
    stop_words = []
    pred_class = []
    right_guesses = 0
    wrong_guesses = 0

    for index, row in data.iterrows():
        words_set = set()
        s = str(row['Review'])
        replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        s = s.translate(replace_punctuation)
        s = s.lower()
        for word in s.split():
            words_set.add(word)
        pos_posterior = 1
        neg_posterior = 1
        words_list = list(words_set)
        # calculate posterior probabilities
        for word in bag_of_words.keys():
            if word in words_list:
                pos_posterior *= bag_of_words[word]['pos']
                neg_posterior *= bag_of_words[word]['neg']
            else:
                pos_posterior *= 1 - bag_of_words[word]['pos']
                neg_posterior *= 1 - bag_of_words[word]['neg']
        # calculate Pr('pos'|review) and Pr('neg'|review)
        pr_pos_as_review = 0
        pr_neg_as_review = 0
        if ((pos_posterior * pos_prob) + (neg_posterior * neg_prob)) != 0:
            pr_pos_as_review = (pos_posterior * pos_prob) / ((pos_posterior * pos_prob) + (neg_posterior * neg_prob))
            pr_neg_as_review = (neg_posterior * neg_prob) / ((pos_posterior * pos_prob) + (neg_posterior * neg_prob))
        # predict class of review
        if pr_pos_as_review > pr_neg_as_review:
            pred_class.append('pos')
            if row['Label'] == 'pos':
                right_guesses += 1
            else:
                wrong_guesses += 1
        else:
            pred_class.append('neg')
            if row['Label'] == 'neg':
                right_guesses += 1
            else:
                wrong_guesses += 1
    # calculate accuracy of classifier
    print('accuracy : ', (right_guesses / (right_guesses + wrong_guesses)))
    return pred_class


def create_prior_prob_without_smoothness(bag_of_words, pos_labels, neg_labels):
    for word in bag_of_words.keys():
        bag_of_words[word]['pos'] /= pos_labels
        bag_of_words[word]['neg'] /= neg_labels
    return bag_of_words


def create_prior_prob_with_laplace(bag_of_words, pos_labels,
                                   neg_labels, words_in_pos_labels, words_in_neg_labels, alpha):
    for word in bag_of_words.keys():
        bag_of_words[word]['pos'] = (bag_of_words[word]['pos'] + alpha) / (pos_labels + (alpha * words_in_pos_labels))
        bag_of_words[word]['neg'] = (bag_of_words[word]['neg'] + alpha) / (neg_labels + (alpha * words_in_neg_labels))
    return bag_of_words


def test_1():
    words_count, pos_labels, neg_labels, words_in_pos_labels, words_in_neg_labels = pre_processing()
    prior_prob = create_prior_prob_without_smoothness(words_count, pos_labels, neg_labels)
    pos_prob = pos_labels / (pos_labels + neg_labels)
    neg_prob = neg_labels / (pos_labels + neg_labels)
    classify(prior_prob, pos_prob, neg_prob)


def test_2():
    bag_of_words, pos_labels, neg_labels, words_in_pos_labels, words_in_neg_labels = pre_processing()
    prior_prob = create_prior_prob_with_laplace(bag_of_words, pos_labels, neg_labels, words_in_pos_labels,
                                                words_in_neg_labels, 1)
    pos_prob = pos_labels / (pos_labels + neg_labels)
    neg_prob = neg_labels / (pos_labels + neg_labels)
    classify(prior_prob, pos_prob, neg_prob)


if __name__ == '__main__':
    print('test 1 (without smoothness)')
    test_1()
    print('test 2 (with laplace smoothness)')
    test_2()
