from classes_new import Corpus, Token, Sentence, Document, Topic, EventMention

import json
import spacy
import argparse
import os
from tqdm import tqdm
import sys
import _pickle as cPickle

from create_elmo_embeddings import *

import torch

nlp = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser(
    description='Feature extraction (predicate-argument structures,'
    'mention heads, and ELMo embeddings)')

parser.add_argument('--config_path',
                    type=str,
                    help=' The path to the configuration json file')
parser.add_argument(
    '--output_path',
    type=str,
    help=' The path to output folder (Where to save the processed data)')

args = parser.parse_args()

if args.config_path != None:

    out_dir = args.output_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.config_path, 'r') as js_file:
        config_dict = json.load(js_file)

    with open(os.path.join(args.output_path, 'build_features_config.json'),
              "w") as js_file:
        json.dump(config_dict, js_file, indent=4, sort_keys=True)

def load_elmo_embeddings(dataset, elmo_embedder, set_pred_mentions):
    '''
    Sets the ELMo embeddings for all the mentions in the split
    :param dataset: an object represents a split (train/dev/test)
    :param elmo_embedder: a wrapper object for ELMo model of Allen NLP
    :return:
    '''
    for (topic_id, topic) in tqdm(dataset.topics.items()):
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                set_elmo_embeddings_to_mentions(elmo_embedder, sent,
                                                set_pred_mentions)

def set_elmo_embeddings_to_mentions(elmo_embedder, sentence,
                                    set_pred_mentions):
    '''
     Sets the ELMo embeddings for all the mentions in the sentence
    :param elmo_embedder: a wrapper object for ELMo model of Allen NLP
    :param sentence: a sentence object
    '''
    avg_sent_embeddings = elmo_embedder.get_elmo_avg(sentence)
    event_mentions = sentence.gold_event_mentions
    entity_mentions = sentence.gold_entity_mentions

    for event in event_mentions:
        set_elmo_embed_to_mention(event, avg_sent_embeddings)

    for entity in entity_mentions:
        set_elmo_embed_to_mention(entity, avg_sent_embeddings)

    # Set the contextualized vector also for predicted mentions
    if set_pred_mentions:
        event_mentions = sentence.pred_event_mentions
        entity_mentions = sentence.pred_entity_mentions

        for event in event_mentions:
            set_elmo_embed_to_mention(
                event,
                avg_sent_embeddings)  # set the head contextualized vector
            print(f"Last one: {event.head_elmo_embeddings}")

        for entity in entity_mentions:
            set_elmo_embed_to_mention(
                entity,
                avg_sent_embeddings)  # set the head contextualized vector

def set_elmo_embed_to_mention(mention, sent_embeddings):
    '''
    Sets the ELMo embeddings of a mention
    :param mention: event/entity mention object
    :param sent_embeddings: the embedding for each word in the sentence produced by ELMo model
    :return:
    '''
    head_index = mention.get_head_index()
    head_embeddings = sent_embeddings[int(head_index)]
    mention.head_elmo_embeddings = torch.from_numpy(head_embeddings)


def find_head(x):
    '''
    This function finds the head and head lemma of a mention x
    :param x: A mention object
    :return: the head word and
    '''

    x_parsed = nlp(x)
    for tok in x_parsed:
        if tok.head == tok:
            if tok.lemma_ == u'-PRON-':
                return tok.text, tok.text.lower()
            return tok.text, tok.lemma_

def order_docs_by_topics(docs):
    '''
    Gets list of document objects and returns a Corpus object.
    The Corpus object contains Document objects, ordered by their gold topics
    :param docs: list of document objects
    :return: Corpus object
    '''
    #Since in the dataset no topics are present add everything as unique
    corpus = Corpus()
    for doc_id, doc in docs.items():
        if doc_id not in corpus.topics:
            topic = Topic(doc_id)
            corpus.add_topic(doc_id, topic)
        topic = corpus.topics[doc_id]
        topic.add_doc(doc_id, doc)
    return corpus

def load_ecb_gold_data(split_txt_file, events_json, entities_json):
    '''
    This function loads the texts of each split and its gold mentions, create document objects
    and stored the gold mentions within their suitable document objects
    :param split_txt_file: the text file of each split is written as 5 columns (stored in data/intermid)
    :param events_json: a JSON file contains the gold event mentions
    :param entities_json: a JSON file contains the gold event mentions
    :return:
    '''
    docs = load_ECB_plus(split_txt_file)
    load_gold_mentions(docs, events_json, entities_json)
    return docs

def load_gold_mentions(docs, events_json, entities_json=None):
    '''
    A function loads gold event and entity mentions
    :param docs: set of document objects
    :param events_json:  a JSON file contains the gold event mentions (of a specific split - train/dev/test)
    :param entities_json: a JSON file contains the gold entity mentions (of a specific split - train/dev/test)
    '''
    load_mentions_from_json(events_json,
                            docs,
                            is_event=True,
                            is_gold_mentions=True)

def load_mentions_from_json(mentions_json_file, docs, is_event,
                            is_gold_mentions):
    '''
    Loading mentions from JSON file and add those to the documents objects
    :param mentions_json_file: the JSON file contains the mentions
    :param docs:  set of document objects
    :param is_event: a boolean indicates whether the function extracts event or entity mentions
    :param is_gold_mentions: a boolean indicates whether the function extracts gold or predicted
    mentions
    '''
    print("Processing mentions...")
    with open(mentions_json_file, 'r') as js_file:
        js_mentions = json.load(js_file)

    for js_mention in tqdm(js_mentions):

        doc_id = js_mention["doc_id"].replace('.xml', '')
        sent_id = js_mention["sent_id"]
        tokens_numbers = js_mention["tokens_number"]
        mention_type = js_mention["mention_type"]
        is_singleton = js_mention["is_singleton"]
        is_continuous = js_mention["is_continuous"]
        mention_str = js_mention["tokens_str"]
        coref_chain = js_mention["coref_chain"]
        if mention_str is None:
            print(js_mention)
        head_text, head_lemma = find_head(mention_str)
        score = js_mention["score"]
        try:
            token_objects = docs[doc_id].get_sentences(
            )[sent_id].find_mention_tokens(tokens_numbers)
        except:
            print('error when looking for mention tokens')
            print('doc id {} sent id {}'.format(doc_id, sent_id))
            print('token numbers - {}'.format(str(tokens_numbers)))
            print('mention string {}'.format(mention_str))
            print('sentence - {}'.format(
                docs[doc_id].get_sentences()[sent_id].get_raw_sentence()))
            raise

        # Sanity check - check if all mention's tokens can be found
        if not token_objects:
            print('Can not find tokens of a mention - {} {} {}'.format(
                doc_id, sent_id, tokens_numbers))

        # Mark the mention's gold coref chain in its tokens
        if is_gold_mentions:
            for token in token_objects:
                if is_event:
                    token.gold_event_coref_chain.append(coref_chain)
                else:
                    token.gold_entity_coref_chain.append(coref_chain)

        if is_event:
            mention = EventMention(doc_id, sent_id, tokens_numbers,
                                   token_objects, mention_str, head_text,
                                   head_lemma, is_singleton, is_continuous,
                                   coref_chain)
        '''
        else:
            mention = EntityMention(doc_id, sent_id, tokens_numbers,
                                    token_objects, mention_str, head_text,
                                    head_lemma, is_singleton, is_continuous,
                                    coref_chain, mention_type)
        '''

        mention.probability = score  # a confidence score for predicted mentions (if used), set gold mentions prob to 1.0
        if is_gold_mentions:
            docs[doc_id].get_sentences()[sent_id].add_gold_mention(
                mention, is_event)
        else:
            docs[doc_id].get_sentences()[sent_id]. \
                add_predicted_mention(mention, is_event,
                                      relaxed_match=False) #relaxed_match = config_dict["relaxed_match_with_gold_mention"]


def load_ECB_plus(processed_ecb_file):
    doc_changed = True
    sent_changed = True
    docs = {}
    last_doc_name = None
    last_sent_id = None

    for line in open(processed_ecb_file, 'r'):
        stripped_line = line.strip()
        try:
            if stripped_line:
                doc_id, sent_id, token_num, word, coref_chain = stripped_line.split(
                    '\t')
                doc_id = doc_id.replace('.xml', '')
        except:
            row = stripped_line.split('\t')
            clean_row = []
            for item in row:
                if item:
                    clean_row.append(item)
            doc_id, sent_id, token_num, word, coref_chain = clean_row
            doc_id = doc_id.replace('.xml', '')

        if stripped_line:
            sent_id = int(sent_id)

            if last_doc_name is None:
                last_doc_name = doc_id
            elif last_doc_name != doc_id:
                doc_changed = True
                sent_changed = True
            if doc_changed:
                new_doc = Document(doc_id)
                docs[doc_id] = new_doc
                doc_changed = False
                last_doc_name = doc_id

            if last_sent_id is None:
                last_sent_id = sent_id
            elif last_sent_id != sent_id:
                sent_changed = True
            if sent_changed:
                new_sent = Sentence(sent_id)
                sent_changed = False
                new_doc.add_sentence(sent_id, new_sent)
                last_sent_id = sent_id

            new_tok = Token(token_num, word, '-')
            new_sent.add_token(new_tok)

    return docs

def combine_datasets(datasets):
    combined = {}
    for dataset in datasets:
        for k, v in dataset.items():
            if k in combined:
                print("Key Collision")
                sys.exit()
            else:
                combined[k] = v
    return combined

def load_data(config_dict):
    dataset = []

    dataset.append(
        load_ecb_gold_data(config_dict["train_text_file"],
                           config_dict["train_event_mentions"],
                           config_dict["train_entity_mentions"]))

    dataset = combine_datasets(dataset)

    return dataset

def main(output_path):

    data = load_data(config_dict)
    dataset = order_docs_by_topics(data)

    if config_dict["load_elmo"]:  # load ELMo embeddings
        elmo_embedder = ElmoEmbedding(config_dict["options_file"],
                                      config_dict["weight_file"])

        logger.info("Loading ELMO embeddings...")
        print("Loading ELMO embeddings...")
        load_elmo_embeddings(dataset, elmo_embedder, set_pred_mentions=False)

    print("Saving data...")

    if output_path == None:
        with open(os.path.join(args.output_path, 'new_dataset'), 'wb') as f:
            cPickle.dump(dataset, f)
    else:
        with open(output_path, 'wb') as f:
            cPickle.dump(dataset, f)


def start_script(config_path, train_text_file, train_event_mentions, output_path):
    global config_dict

    with open(config_path, 'r') as js_file:
        config_dict = json.load(js_file)

    config_dict['train_text_file'] = train_text_file
    config_dict['train_event_mentions'] = train_event_mentions

    main(output_path)


if __name__ == '__main__':

    main(None)




