import re

# Text cleaning and tokenization

if __name__ == '__main__':
    sentence = 'Sunil tweeted, "Witnessing 70th Republic Day of India from Rajpath, \
New Delhi. Mesmerizing performance byh Indian Army! Awesome airshow! @india_official \
#India #70thRepublic_Day. For more photos ping me sunil@photoking.com :)'
    
    tokens = re.sub(r'([^\s\w]|_)+', ' ', sentence).split()
    """
    (Capturing group #1
        [^ Negated set.
            \s whitespace. Matches any whitespace character (spaces, tabs, line break).
            \w word. Matches any word character (alphanumeric & underscore) 
        ]
        | Alternation. Acts like a boolean OR.
        _ Character. Matches a "_" character.
    )
    + Quantifier. Match 1 or more of preceding token.
    """
    print(sentence)
    print(tokens)
