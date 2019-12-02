import json
levels = ['A1', 'A2', 'B1', 'B2', 'C1']

def parse(path):
    word_to_difficulty = {}
    with open(path, 'r') as f:
        level = None
        for line in f:
            line = line.strip()
            if line in levels:
                level = line
                continue
            words = line.split(' ')
            if len(words) == 0 or words[0] == '':
                continue
            if 'Oxford' in words or '/' in words:
                continue
            if level == None:
                print('oh no! level not parsed')
                print(words)
                continue
            word = words[0]
            if word[-1].isnumeric():
                word = word[:-1]
            if not word.isalpha():
                print('warning: %s, %s' % (word, line))
            word_to_difficulty[word] = level
    return word_to_difficulty

vocabularies = parse('oxford-3000-by-level.txt')
vocabularies.update(parse('oxford-5000-by-level.txt'))
del vocabularies['adj.']
del vocabularies['a, an']
vocabularies['a'] = 'A1'
vocabularies['an'] = 'A1'

print(len(vocabularies))
json.dump(vocabularies, open('words_by_difficulty.json', 'w'))
