from sklearn.datasets import fetch_20newsgroups
import re


# this function removes noise from the dataset
# the raw newsgroup posts contain headers, quoted replies and signatures
# these things dont really contribute to semantic meaning so we remove them
def clean_text(text):

    # remove email headers
    text = re.sub(r'From:.*\n', '', text)
    text = re.sub(r'Subject:.*\n', '', text)
    text = re.sub(r'Organization:.*\n', '', text)

    # remove quoted replies
    text = re.sub(r'>.*\n', '', text)

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# loads the 20 newsgroups dataset and applies cleaning
def load_dataset():

    # sklearn provides a convenient loader for this dataset
    dataset = fetch_20newsgroups(remove=("headers", "footers", "quotes"))

    docs = []

    for doc in dataset.data:

        cleaned = clean_text(doc)

        # extremely short posts dont produce meaningful embeddings
        # so we discard them
        if len(cleaned.split()) > 20:
            docs.append(cleaned)

    return docs