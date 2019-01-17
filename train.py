from string import punctuation
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader

def train(FLAGS):
    # download the files if needed
    print ('[INFO]Checking if the data is present...')
    if os.path.exists(FLAGS.dataset + 'labels.txt') and
        os.path.exists(FLAGS.dataset + 'reviews.txt'):
        print ('[INFO]Files not found!')
        print ('[INFO]Starting to download files...')
        subprocess.call(['./dataset/download.sh'])
        print ('[INFO]Files downloaded successfully!')
    else:
        print ('[INFO]Files found!!')

    # read the data
    print ('[INFO]Reading the datasets...')
    with open(FLAGS.dataset + 'reviews.txt', 'r') as f:
        reviews = f.read()
    with open(FLAGS.dataset + 'labels.txt', 'r') as f:
        labels = f.read()
    print ('[INFO]Dataset read!')

    # preprocess data
    features = preprocess(reviews, labels, FLAGS.seq_length)

    # Split the data
    # Get the split fraction
    split_frac = FLAGS.split_frac

    # Get the data for training set
    tr_idx = int(len(features) * split_frac)
    train_x, train_y = features[: tr_idx], np.array(encoded_labels[: tr_idx])

    # Get the data for validation set
    va_idx = tr_idx + int(len(features[tr_idx : ]) * 0.5)
    val_x, val_y = features[tr_idx : va_idx], np.array(encoded_labels[tr_idx : va_idx])

    # Get the test data
    test_x, test_y = features[va_idx : ], np.array(encoded_labels[va_idx : ])

    # Create DataLoaders
    ## Create the TensorDatasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    ## DataLoaders
    bs = FLAGS.batch_size
    train_loader = DataLoader(train_data, shuffle=True, batch_size=bs)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=bs)
    test_loader = DataLoader(valid_data, shuffle=False, batch_size=bs)


def preprocess(reviews, labels, seq_length):
    # Making all the characters lowercase to ease for model understanding
    reviews = reviews.lower()

    # Getting rid of all the punctuations in the reviews
    all_text = ''.join([c for c in reviews if c not in punctuation])

    # Create reviews list by splitting with newlines
    reviews_split = all_text.split('\n')

    # Join all reviews with a ' '
    all_text = ' '.join(reviews_split)

    # Now create a list of words by splitting by ' ' to create the vocabulary
    words = all_text.split()

    # Build a dictionary that maps words to integers
    # Count occurance of each word
    vocab_count = Counter(words)
    # Sort the vocab in the decreasing order of the times each word is used
    vocab = sorted(vocab_count, key=vocab_count.get, reverse=True)
    # Assign integers to each word
    vocab_to_int = {word : ii for ii, word in enumerate(vocab, 1)}

    # Tokenize each review
    reviews_ints = []
    for review in reviews_split:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])

    print ('[INFO] Unique words in the vocab: {}'.format(len(vocab_to_int)))

    # Encoding the labels
    encoded_labels = [1 if label == 'positive' else 0 for label in labels.split('\n')]

    # Remove outliers
    # Removing 0 length reviews
    # Getting the idxs for non 0 length reviews
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
    # Removing 0 length reviews
    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    encoded_labels = [encoded_labels[ii] for ii in non_zero_idx]

    print ('[INFO]Number of reviews left after removing 0 length reviews {}'.format(len(non_zero_idx)))

    # Pad the features
    features = pad_features(reviews_ints, seq_length)

    # A few tests to make life easy
    assert len(features) == len(reviews_ints)
    assert len(features[0]) == seq_length

    return features, encoded_labels


def pad_features(reviews_ints, seq_length):
    """
    Return features of reviews_ints, where each review is padded with 0s or truncated
    to the input seq_length
    """
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for ii, rint in enumerate(reviews_ints):
        features[ii, -len(rint) : ] = rint[: seq_length]

    return features
