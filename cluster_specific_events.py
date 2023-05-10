# This file combines the necessary steps to generate the datafiles

# Keep in mind that an article could belong to 2 events so the system may run multiple times in that case
from ast import literal_eval
import pandas as pd
from src.data.convert_format import generate_corpus, generate_mention_json
from src.features.build_features_new import start_script
from src.features import build_features_new

specific_event = 'http://www.wikidata.org/entity/Q104218016'
event_name = specific_event.split('/')[-1]

if __name__ == '__main__':

    data = pd.read_csv('data/new_data/final_data_with_predictions_new.csv', index_col=0) #Read data, split the events
    data.Event = data.Event.apply(literal_eval)
    data = data.explode('Event')

    data = data[data['Event'] == specific_event]

    corpus_output = f'generated_data/corpora/{event_name}_corpus.txt'
    mention_output = f'generated_data/mentions/{event_name}_event_mentions.json'
    dataset_output = f'generated_data/datasets/{event_name}_dataset'

    generate_corpus(data, corpus_output) #get the first two files
    generate_mention_json(data, mention_output)

    build_features_new.start_script('build_features_config.json', corpus_output, mention_output, dataset_output)






