import pandas as pd
import random
from nltk.corpus import wordnet as wn
from nlpaug.augmenter.word.back_translation import BackTranslationAug

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def replace_with_synonym(sentence):
    words = sentence.split()
    idx = random.randint(0, len(words) - 1)
    word = words[idx]
    synonyms = get_synonyms(word)
    if synonyms:
        synonym = random.choice(synonyms)
        words[idx] = synonym
    return ' '.join(words)

def balance_classes_with_augmentation(dataFrame, minority_classes, majority_class):
    df = dataFrame.copy()
    augmented_data = []

    all_augmented_sentences = set()

    minority_data = df[df['Labels'].isin(minority_classes)]
    majority_data = df[df['Labels'] == majority_class]
    max_samples_class = len(majority_data)

    back_translation = BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-de',
        to_model_name='Helsinki-NLP/opus-mt-de-en',
        device='cuda',
        batch_size=10,
    )

    for label in minority_classes:
        minority_class_data = minority_data[minority_data['Labels'] == label]
        samples_needed = max_samples_class - len(minority_class_data)

        if samples_needed > 0:
            augmented_sentences = minority_class_data['Sentences'].tolist()
            while len(augmented_sentences) < samples_needed:
                print(len(augmented_sentences), label, samples_needed)
                augmented = back_translation.augment(augmented_sentences, n=samples_needed - len(augmented_sentences))
                augmented = [replace_with_synonym(sentence) for sentence in augmented if sentence not in all_augmented_sentences]
                augmented_sentences.extend(augmented)
                all_augmented_sentences.update(augmented)

            augmented_sentences = augmented_sentences[:samples_needed]

            augmented_df = pd.DataFrame({
                'Sentences': augmented_sentences,
                'Labels': label
            })

            augmented_data.append(augmented_df)

            df = pd.concat([df, augmented_df], ignore_index=True)

    return df

