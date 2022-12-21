import json

with open("processed/all_data.json", "r") as f:
    data = json.load(f)

"""
print(data[0])
print(' '.join(data[0]['text']))

for i, (x,y) in enumerate(zip(data[0]['text'], data[0]['label'])):
    if y == 'break':
        data[0]['text'].insert(i, 'break')

print(' '.join(data[0]['text']))
"""

for d in data:
    d['phoneme_label'] = d['phoneme']
    d['text_label'] = d['text']
    d['phoneme'] = ' '.join(d['phoneme'])
    d['text'] = ' '.join(d['text'])
    for i, (x,y) in enumerate(zip(d['phoneme_label'], d['label'])):
        if y == 'break':
            d['phoneme_label'].insert(i, 'break')
            d['text_label'].insert(i, 'break')
    assert len(d['phoneme_label']) == len(d['text_label'])
    d['phoneme_label'] = ' '.join(d['phoneme_label'])
    d['text_label'] = ' '.join(d['text_label'])
    del d['label']

with open("processed/seq2seq_all_data.json", "w", encoding='UTF-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)