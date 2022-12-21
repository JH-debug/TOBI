# from lightning_transformers.core import TransformerDataModule
import os
import pandas as pd

text_path = "script/"
label_path = "lab/"

file_name = []
text = []
label = []

for file in sorted(os.listdir(text_path)):
    filename, file_extension = os.path.splitext(os.path.basename(file))
    if file_extension == '.txt':
        with open(text_path + file, 'r') as f:
            file_name.append(filename)
            text.append(f.read().split('\n')[0])

for file in sorted(os.listdir(label_path)):
    filename, file_extension = os.path.splitext(os.path.basename(file))
    if file_extension == '.btl':
        with open(label_path + file, 'r') as f:
            label_only = [x for x in f.read().split() if x == '*' or x == ';']
            """
            if label_only[-1] == ';':
                label_only[-1] = '*'

            for i, l in enumerate(label_only):
                if l == ';':
                    del label_only[i-1]
            """
            label.append(label_only)


assert len(text) == len(label)

"""
for x, y in zip(text, label):
    assert len(x.split()) == len(y)

df = pd.DataFrame({'filename': file_name, 'text': text, 'label': label})
df.to_csv('data.csv', index=False)
"""

print(text[1])
print(label[1])
print(len(text[1].split()), len(label[1]))
if label[1][-1] == ';':
    label[1][-1] = '*'
print(label[1])
for i, l in enumerate(label[1]):
    if l == ';':
        del label[1][i - 1]
print(label[1])
print(len(text[1].split()), len(label[1]))
