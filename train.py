from string import punctuation
from collections import Counter

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
    preprocess(reviews, labels)

def preprocess(reviews, labels):
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
    
