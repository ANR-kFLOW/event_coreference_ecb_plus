# This file combines the necessary steps to generate the datafiles

# Keep in mind that an article could belong to 2 events so the system may run multiple times in that case
from ast import literal_eval
import pandas as pd

from convert_format import generate_corpus, generate_mention_json
import build_features_new
import os



specific_event = None

if __name__ == '__main__':

    data = pd.read_csv('data/new_data/final_data_with_predictions_new.csv', index_col=0) #Read data, split the events
    data.Event = data.Event.apply(literal_eval)
    data = data.explode('Event')

    import os

    if not os.path.exists('generated_data/corpora/'):
        os.makedirs('generated_data/corpora/')

    if not os.path.exists('generated_data/mentions/'):
        os.makedirs('generated_data/mentions/')

    if not os.path.exists('generated_data/datasets/'):
        os.makedirs('generated_data/datasets/')

    if specific_event != None:

        event_name = specific_event.split('/')[-1]

        data = data[data['Event'] == specific_event]

        corpus_output = f'generated_data/corpora/{event_name}_corpus.txt'
        mention_output = f'generated_data/mentions/{event_name}_event_mentions.json'
        dataset_output = f'generated_data/datasets/{event_name}_dataset'

        generate_corpus(data, corpus_output) #get the first two files
        generate_mention_json(data, mention_output)

        build_features_new.start_script('build_features_config.json', corpus_output, mention_output, dataset_output)

    else: #Build the dataset for all events seperatly

        for event in data['Event'].unique():
            print(event)

            sub_data = data[data['Event'] == event]
            event_name = event.split('/')[-1]

            corpus_output = f'generated_data/corpora/{event_name}_corpus.txt'
            mention_output = f'generated_data/mentions/{event_name}_event_mentions.json'
            dataset_output = f'generated_data/datasets/{event_name}_dataset'

            generate_corpus(sub_data, corpus_output)  # get the first two files
            generate_mention_json(sub_data, mention_output)

            build_features_new.start_script('build_features_config.json', corpus_output, mention_output, dataset_output)




