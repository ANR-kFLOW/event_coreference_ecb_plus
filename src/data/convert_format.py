import spacy
import pandas as pd
from tqdm import tqdm
import json


def generate_corpus(dataset, output_path = 'data/new_data/dataset_corpus.txt'):
    """
    This generates the corpus text file from the dataset.
    This 1 out of the 2 files needed for the build feature step
    :param dataset: The dataset containing the sentences, sentence number and document name
    """
    print("Generating corpus...")
    nlp = spacy.load("en_core_web_sm")

    out_file = open(output_path, 'w')
    for  index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        for i, token in enumerate(nlp(row['Sentence'])):
            out_file.write(f"{row['URI']}\t{row['Sentence_num']}\t{i}\t{token}\t-\n")
        out_file.write('\n')

def generate_mention_json(dataset, output_path = 'data/new_data/dataset_event_mentions.json'):
    """
    This converts the dataset into the mention file.
    It extracts the subject and object from the file
    :param dataset: The dataset containing the sentence number, document name, subject and object
    """

    print("Generating mention json file...")

    total_mention_data= [] #This will contain all the mentions
    nlp = spacy.load("en_core_web_sm")

    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):

        if not row.isnull().values.any():

            mentions = [row['Subject'], row['Object']]

            for mention in mentions:

                token_mention = nlp(mention) #Split the subject/ object into tokens
                tokenized_sentence = nlp(row['Sentence'])

                matcher = spacy.matcher.PhraseMatcher(nlp.vocab)
                matcher.add("MENTION", [token_mention])

                matches = matcher(tokenized_sentence)

                spans= [tokenized_sentence[start:end] for _, start, end in matches]

                if len(spans) != 0:
                    span= spacy.util.filter_spans(spans)[0] #Take only the first best match

                    token_numbers = [x for x in range(span.start, span.end)]
                    token_text = span.text

                    mention_data = {"coref_chain": "-",
                                    "doc_id": row['URI'],
                                    "is_continuous": True,
                                    "is_singleton": False,
                                    "mention_type": "ACT",
                                    "score": -1.0,
                                    "sent_id": row['Sentence_num'],
                                    "tokens_number": token_numbers,
                                    "tokens_str": token_text
                                    }  # Dictionary of the mention

                    total_mention_data.append(mention_data)

    json_object = json.dumps(total_mention_data, indent=4)

    with open(output_path, "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    dataset = pd.read_csv('data/final_data_with_predictions.csv')
    #generate_corpus(dataset)
    generate_mention_json(dataset)



