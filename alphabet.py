import collections
import string


def alphabet_factory():
    char_blank = "*"
    char_space = " "
    char_apostrophe = "'"
    labels = char_blank + char_space + char_apostrophe + string.ascii_lowercase
    alphabet = Alphabet(char_blank, char_space, labels)
    return alphabet


class Alphabet:
    """Maps characters to integers and vice versa"""

    def __init__(self, char_blank, char_space, labels):
        self.char_space = char_space
        self.char_blank = char_blank

        labels = list(labels)
        self.length = len(labels)
        enumerated = list(enumerate(labels))
        flipped = [(sub[1], sub[0]) for sub in enumerated]

        d1 = collections.OrderedDict(enumerated)
        d2 = collections.OrderedDict(flipped)
        self.mapping = {**d1, **d2}

    def __len__(self):
        return self.length

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        if isinstance(text, list):
            return [self.text_to_int(i) for i in text]
        else:
            return [self.mapping[i] + self.mapping[self.char_blank] for i in text]

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        if len(labels) > 0 and isinstance(labels[0], list):
            return [self.int_to_text(label) for label in labels]
        else:
            string = [self.mapping[i] for i in labels]
            string = ''.join(string).replace(self.char_blank, ' ')
            return string
