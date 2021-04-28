import nltk
import sys

from nltk.util import pr

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | VP | NP

NP -> N | Det N | NP PP | Det AdjP N
VP -> V | V NP | V PP | Adv VP | VP Adv

PP -> P NP
AdjP -> Adj | Adj AdjP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # Initialize empty words list
    words = []
    
    # Iterate over words in tokenized sentence
    for word in nltk.tokenize.word_tokenize(sentence):
        # If word contains at least one alphabetic character
        if any([char.isalpha() for char in word]):
            # Add lowercased word to words list
            words.append(word.lower())
    
    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # Initialize empty NPs chunks list
    nps_chunks = []
    
    # Iterate over all NPs in the sentence tree
    for NP in tree.subtrees(lambda t: t.label() == "NP"):
        # If NP doesn't contain any other NPs as subtrees
        if len(list(NP.subtrees(lambda l: l.label() == "NP"))) == 1:
            # Add it to NPs chunks list
            nps_chunks.append(NP)
        
    return nps_chunks


if __name__ == "__main__":
    main()
