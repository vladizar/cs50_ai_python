import os
import random
import re
import sys

from numpy.random import choice

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Initialize PageRank dictionary
    pages_rank = dict()
    pages_num = len(corpus)
    links_num = len(corpus[page])
    
    # If page has links
    if links_num:
        # Each page (excluding current) has equal chance to get visited because of damping factor
        random_link_chance = (1 - damping_factor) / pages_num
        
        # Some pages have increased chance of being visited because of links on current page
        child_link_chance = damping_factor / links_num
        
        # Sum up probabilities for each page (excluding current)
        for link in corpus.keys():
            pages_rank[link] = random_link_chance
        
        for link in corpus[page]:
            pages_rank[link] += child_link_chance
    # If page hasn't got any links
    else:
        # All pages (including current) have equal chances
        random_link_chance = 1 / pages_num
        
        for page in corpus.keys():
            pages_rank[page] = random_link_chance

    return pages_rank

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize PageRank dictionary
    pages_rank = dict()
    
    # Choose first page to visit by random (with equal chances)
    pages_chain = [choice(list(corpus.keys()))]
    
    # Simulate 'n' random surfer pages visits according to 'transition_model' probabilities
    for i in range(n - 1):
        current_page = pages_chain[i]
        pages_chance = transition_model(corpus, current_page, damping_factor)
        pages_chain.append(choice(list(pages_chance.keys()), p=list(pages_chance.values())))
        
    # Find out PageRank of each page using the 'pages_chain' random surfer visited
    for page in corpus.keys():
        page_chance = pages_chain.count(page) / n
        pages_rank[page] = page_chance
        
    return pages_rank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize PageRank dictionary
    pages_rank = dict()
    N = len(corpus) # Number of pages in corpus
    precision = 4 # Precision of PageRank values
    
    # Start by assigning each page an equal PageRank
    for page in corpus.keys():
        pages_rank[page] = 1 / N
        
    # While changes are valuable according to 'precision'
    changes = True
    while changes:
        changes = False
        
        # Iterate over each page, assign it a PageRank using iterative algorithm
        for page1 in pages_rank.keys():
            # Probability of visiting a page by damping_factor
            page_rank = (1 - damping_factor) / N
            
            # Summing up probabilities of visiting a page from other pages
            for page2 in pages_rank.keys():
                # If looking at different pages
                if page2 != page1:
                    # Number of links page has
                    links_num = len(corpus[page2])
                    
                    # If no links on page, interpret it as having one link for every page in the corpus (including itself)
                    if not links_num:
                        links_num = N
                    
                    # If page has links to PageRank calculating page increase PageRank value
                    if page1 in corpus[page2] or links_num == N:
                        page_rank += damping_factor * pages_rank[page2] / links_num
            
            # Check if changes are valuable according to 'precision'
            if round(pages_rank[page1], precision) != round(page_rank, precision):
                changes = True
            
            # Update page PageRank
            pages_rank[page1] = page_rank
    
    return pages_rank


if __name__ == "__main__":
    main()
