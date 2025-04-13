class StopWords:
    def __init__(self):
        self.stop_words = {}

    def add(self, word):
        self.stop_words[word] = 1

    def set_stop_words(self):
        self.stop_words = {}

    def is_stop_word(self, word):
        return word in self.stop_words
