import os

cats = os.listdir('../hier')

for i, cat in enumerate(cats):
    print(f"On {cat} ({i}/{len(cats)}")
    os.system(f'python3 make_dataset.py ../data/{cat} {cat}')
