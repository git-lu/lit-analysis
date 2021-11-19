import os
from pathlib import Path

import pandas as pd
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

TASSSentLabels = ['NEG', 'NEU', 'POS']

TASSEmotionLabels = [
    'others',
    'joy',
    'sadness',
    'anger',
    'surprise',
    'disgust',
    'fear',
]


def parse_sent_label(label):
    parsing = {
        'P': 'POS',
        'N': 'NEG',
        'NEU': 'NEU'
    }
    return parsing[label]


class TASSSentimentData(lit_dataset.Dataset):
    """Loader for TASS text data."""

    def __init__(self, folder_path):
        print(f"cwd: {os.getcwd()}")
        original_path = os.getcwd()
        os.chdir(folder_path)
        self._examples = []
        self._countries = []
        for file in os.listdir():
            df = pd.read_csv(file, sep='\t', names=['id', 'tweet', 'label'])
            country = Path(file).name
            self._countries.append(country)
            for _, row in df.iterrows():
                self._examples.append(
                    {
                        'tweet': row['tweet'],
                        'label': parse_sent_label(row['label']),
                        'country': country
                    }
                )
        os.chdir(original_path)

    def spec(self):
        return {
            'tweet': lit_types.TextSegment(),
            'country': lit_types.CategoryLabel(vocab=TASSSentLabels),
            'label': lit_types.CategoryLabel(vocab=TASSSentLabels)
        }


class TASSEmotionsData(lit_dataset.Dataset):
    """Loader for TASS text data."""

    def __init__(self, path):
        df = pd.read_csv(path, sep='\t')
        self._examples = [{
            'tweet': row['tweet'].strip(),
            'label': row['label '].strip(),
        } for _, row in df.iterrows() if row['label '].strip() in TASSEmotionLabels]

    def spec(self):
        return {
            'tweet': lit_types.TextSegment(),
            'label': lit_types.CategoryLabel(vocab=TASSEmotionLabels)
        }
