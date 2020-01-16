def get_synonyms(model, words):
    for result in words:
        word = result['word']
        num = 1
        positive = []
        while True:
            word_name = word + '_' + str(num)
            if word_name in model:
                positive.append(word_name)
            else:
                break
            num += 1
        if not positive:
            continue
        print({
            'word': word,
            'synonyms': [{
                'key': float(score),
                'desc': item
            } for item, score in model.most_similar(positive=positive)]
        })
