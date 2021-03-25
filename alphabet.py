class Alphabet:
    """Maps characters to integers and vice versa"""
    _FIRST_ASCII_CHAR_IDX = 97
    delimiter = ' '

    def __init__(self):
        self.char_map = {}
        self.index_map = {}

        for idx in range(26):
            ascii_idx = self._FIRST_ASCII_CHAR_IDX + idx
            self.char_map[chr(ascii_idx)] = idx + 2
            self.index_map[idx + 2] = chr(ascii_idx)
        self.char_map["'"] = 0
        self.char_map["<SPACE>"] = 1
        self.index_map[0] = "'"
        self.index_map[1] = "<SPACE>"

    def __len__(self):
        return len(self.char_map.keys()) + 1

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        if len(labels) > 0 and isinstance(labels[0], list):
            return [self.int_to_text(label) for label in labels]
        else:
            string = [self.index_map[i] for i in labels]
            string = ''.join(string).replace('<SPACE>', ' ')
            return string
