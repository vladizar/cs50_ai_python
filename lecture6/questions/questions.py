import nltk
import sys
import os

from nltk.util import pr
from string import punctuation
from math import log

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Get path to directory with files
    path = os.path.join(directory)
    
    # Initialize empty dictionary
    files = dict()
    
    # Iterate over files in directory
    for filename in os.listdir(path):
        # Get file path
        file_path = os.path.join(path, filename)
        
        # Open file
        with open(file_path, "r") as f:
            # Store it's content to dictionary under filename key
            files[filename] = f.read()
    
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Initialize empty words list
    words = list()
    
    # Iterate over potential words in tokenized document
    for word in nltk.tokenize.word_tokenize(document):
        # If word isn't a punctuation
        if not word[0] in punctuation:
            # Lowercase it
            word = word.lower()
            
            # If word isn't an English language stopword
            if not word in nltk.corpus.stopwords.words("english"):
                # Add it to words list
                words.append(word)
    
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Initialize empty IDFs dictionary
    idfs = dict()
    
    # Store documents number
    docs_num = len(documents)
    
    # Iterate over documents
    for document in documents.values():
        # Iterate over words in document
        for word in document:
            # If there is no word IDF yet
            if word not in idfs:
                # Count it's appearences
                appearences = 0
                for doc in documents.values():
                    if doc == document or word in doc:
                        appearences += 1
                
                # Store word IDF value in IDFs dictionary
                idfs[word] = log(docs_num / appearences)
    
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Initialize empty files ranks dictionary
    files_ranks = dict()
    
    # Iterate over files
    for filename in files:
        # Initialize file rank value to 0
        file_rank = 0
        
        # Iterate over words in query
        for word in query:
            # Count it's appearences in file
            appearences = files[filename].count(word)
            
            # If word appeared at least once
            if appearences:
                # Add it's TF-IDF value to file rank value
                file_rank += appearences * idfs[word]
        
        # Store file rank value in files ranks dictionary
        files_ranks[filename] = file_rank
    
    # Sort filenames from files ranks dictionary by their values from bigger to smaller
    filenames = sorted(files_ranks, reverse=True, key=lambda f: files_ranks[f])
    
    # Return first 'n' filenames
    return filenames[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Initialize empty sentences ranks dictionary
    sentences_ranks = dict()
    
    # Iterate over sentences
    for sentence in sentences:
        # Store sentence words list
        sentence_words = sentences[sentence]
        
        # Store all words in query that also appear in the sentence
        valuable_words = [word for word in query if word in sentence_words]
        
        # Compute sentence query term density
        term_density = len(valuable_words) / len(sentence_words)
        
        # Compute sentence rank value using formula
        # 'sum of IDF values for any word in the query that also appears in the sentence' * 10 + 'sentence query term density'
        sentence_rank = sum([idfs[word] for word in valuable_words]) * 10 + term_density
        
        # Store sentence rank value in sentences ranks dictionary
        sentences_ranks[sentence] = sentence_rank
    
    # Sort sentences from sentences ranks dictionary by their values from bigger to smaller
    sentences = sorted(sentences_ranks, reverse=True, key=lambda f: sentences_ranks[f])
    
    # Return first 'n' sentences
    return sentences[:n]


if __name__ == "__main__":
    main()
