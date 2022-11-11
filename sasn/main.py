import argparse
import json
import logging
import os
from pathlib import Path
import pickle
import tempfile
import random
import sys

import nltk
import numpy as np
import torch
from tqdm import tqdm, trange

from config import settings
from components.vocab import VocabGenerator, SOS_token, EOS_token
from components.example import Example
from grammar.action import *
from grammar.grammar import *
from grammar.metaparser import *
from grammar.syntax_tree import *
from grammar.parser_generator import *
from grammar.unparser_generator import *

from model.encoder import Encoder
from model.decoder import Decoder
from model.utils import *

# ============================================================================
# Execution modes
# ============================================================================

def grammargen():
    settings.LOGGER.info('Parsing the grammar...')
    with open(settings.GRAMMAR_PATH, 'r') as grammar_file:
        description = grammar_file.read()
    grammar = parse_grammar(description)
    if settings.LOGGER.getEffectiveLevel() == logging.DEBUG:
        settings.LOGGER.debug('Printing the parser grammar...')
        print_grammar(grammar, settings.LOGHANDLER.stream)
    settings.LOGGER.info('Verifying the parsed grammar...')
    check_grammar(grammar)
    settings.LOGGER.info('Generating the parser...')
    with open(settings.PARSER_PATH, 'w') as parser_file:
        generate_parser(grammar, parser_file)
    settings.LOGGER.info('Generating the unparser...')
    with open(settings.UNPARSER_PATH, 'w') as unparser_file:
        generate_unparser(grammar, unparser_file)
    settings.LOGGER.info('Done!')

# ------------------------------------

def preprocess():
    settings.LOGGER.info('Opening the JSON datafile...')
    JSON_PATH = settings.SELF_PATH.joinpath(settings.JSON)
    with open(JSON_PATH, 'r') as json_file:
        json_data = json.load(json_file)

    intent_vocab_generator = VocabGenerator()
    action_vocab_generator = VocabGenerator()
    example_count = len(json_data)
    settings.LOGGER.info(f'Example count: {example_count}')

    # Create temporary files along with how many examples they can take.
    settings.LOGGER.info('Creating temporary files...')
    files = list()
    for _ in trange(example_count // settings.INMEMORY_SHUFFLE_COUNT + 1):
        temp = tempfile.TemporaryFile(mode='rb+', buffering=0)
        files.append([temp, settings.INMEMORY_SHUFFLE_COUNT])

    # --------------------
    # Preprocess examples.
    # --------------------

    max_intent_length = 0
    max_action_length = 0
    settings.LOGGER.info('Preporcessing examples...')
    for example_json in tqdm(json_data):
        # Preprocess the input and output of an example.
        intent = nltk.word_tokenize(example_json['intent'].lower())
        try:
            syntax_tree = settings.PARSE(example_json['snippet'])
        except settings.PARSER_MODULE.LanguageError as e:
            settings.LOGGER.critical(f"Error at question {example_json['question_id']}:")
            settings.LOGGER.critical(str(e))
            sys.exit(1)
        actions = syntax_tree_to_actions(syntax_tree)
        max_intent_length = max(len(actions), max_intent_length)
        max_action_length = max(len(actions), max_action_length)

        # Distribute the example to a random temporary file.
        chosen = random.choice([f for f in files if f[1]])
        pickle.dump(Example(intent, actions), chosen[0])
        chosen[1] -= 1

        # Update the vocabularies.
        intent_vocab_generator.add(intent)
        action_vocab_generator.add(actions)

    intent_vocab = intent_vocab_generator.generate()
    action_vocab = action_vocab_generator.generate()

    # -------------------
    # Output the results.
    # -------------------

    settings.LOGGER.info('Opening the pickled datafile...')
    PICKLE_PATH = settings.SELF_PATH.joinpath(settings.PICKLE)
    with open(PICKLE_PATH, 'wb') as pickle_file:
        settings.LOGGER.info('Dumping meta information...')
        pickle.dump(intent_vocab, pickle_file)
        pickle.dump(action_vocab, pickle_file)
        pickle.dump(example_count, pickle_file)
        pickle.dump(max_intent_length, pickle_file)
        pickle.dump(max_action_length, pickle_file)

        # Concatenate the files in a random order.
        settings.LOGGER.info('Dumping examples in random order...')
        random.shuffle(files)
        for temp, _ in tqdm(files):
            temp.seek(0)  # Reset the file position to read from it.
            pickle_file.write(temp.read())
        settings.LOGGER.info('Done!')

# ------------------------------------

def train():
    # Load data files and language modules.
    settings.LOGGER.info('Opening the datafile...')
    DATA_PATH = settings.SELF_PATH.joinpath(settings.DATAFILE)
    datafile = open(DATA_PATH, 'rb')
    intent_vocab = pickle.load(datafile)
    action_vocab = pickle.load(datafile)
    example_count = pickle.load(datafile)
    settings.LOGGER.info(f'Example count: {example_count}')
    max_intent_len = pickle.load(datafile)
    max_action_len = pickle.load(datafile)
    settings.LOGGER.info('Parsing the grammar...')
    with open(settings.GRAMMAR_PATH, 'r') as grammar_file:
        description = grammar_file.read()
    grammar = parse_grammar(description)
    if settings.LOGGER.getEffectiveLevel() == logging.DEBUG:
        settings.LOGGER.debug('Printing the parser grammar...')
        print_grammar(grammar, settings.LOGHANDLER.stream)
    settings.LOGGER.info('Verifying the parsed grammar...')
    check_grammar(grammar)

    settings.LOGGER.info('Initializing training...')
    start_position = datafile.tell()

    # Initialize the encoder.
    encoder = Encoder(len(intent_vocab), settings.HIDDEN_SIZE)
    encoder.to(settings.DEVICE)
    encoder.train()
    encoder_optimizer = torch.optim.SGD(
        encoder.parameters(), lr=settings.LEARNING_RATE)

    # Initialize the decoder.
    decoder = Decoder(settings.HIDDEN_SIZE, len(action_vocab))
    decoder.to(settings.DEVICE)
    decoder.train()
    decoder_optimizer = torch.optim.SGD(
        decoder.parameters(), lr=settings.LEARNING_RATE)

    criterion = torch.nn.NLLLoss()

    # -------------------
    # Main training loop.
    # -------------------

    print_every = int(example_count * settings.PRINT_EVERY)

    settings.LOGGER.info('Training...')
    for epoch in trange(settings.MAX_EPOCH):
        datafile.seek(start_position)  # Reset file position.

        print_loss_total = 0

        for iteration in range(example_count):
            example = pickle.load(datafile)
            intent_tensor = tensor_from_sequence(intent_vocab, example.intent)
            action_tensor = tensor_from_sequence(action_vocab, example.actions)

            # ------------------------

            encoder_hidden = encoder.init_hidden()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs = torch.zeros(
                max_action_len, settings.HIDDEN_SIZE, device=settings.DEVICE)

            loss = 0

            for ei in range(intent_tensor.size(0)):
                encoder_output, encoder_hidden = encoder(intent_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=settings.DEVICE)
            decoder_hidden = encoder_hidden

            for di in range(action_tensor.size(0)):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += criterion(decoder_output, action_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            loss = loss.item() / action_tensor.size(0)

            # ------------------------

            print_loss_total += loss

        if print_every and epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            settings.LOGGER.debug(f'{print_loss_avg}')

    # Teardown.
    datafile.close()
    settings.LOGGER.info(f'Saving model...')
    MODEL_PATH = settings.SELF_PATH.joinpath(settings.MODEL)
    torch.save({'max_action_len': max_action_len,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'intent_vocab': intent_vocab,
                'action_vocab': action_vocab}, MODEL_PATH)
    settings.LOGGER.info(f'Done!')

# ------------------------------------

def query():
    # Load language models.
    settings.LOGGER.info('Parsing the grammar...')
    with open(settings.GRAMMAR_PATH, 'r') as grammar_file:
        description = grammar_file.read()
    grammar = parse_grammar(description)
    if settings.LOGGER.getEffectiveLevel() == logging.DEBUG:
        settings.LOGGER.debug('Printing the parser grammar...')
        print_grammar(grammar, settings.LOGHANDLER.stream)
    settings.LOGGER.info('Verifying the parsed grammar...')
    check_grammar(grammar)

    settings.LOGGER.info('Initializing inference...')
    MODEL_PATH = settings.SELF_PATH.joinpath(settings.MODEL)
    checkpoint = torch.load(MODEL_PATH)
    max_action_len = checkpoint['max_action_len']
    intent_vocab = checkpoint['intent_vocab']
    action_vocab = checkpoint['action_vocab']

   # Initialize the encoder.
    encoder = Encoder(len(intent_vocab), settings.HIDDEN_SIZE)
    encoder.to(settings.DEVICE)
    settings.LOGGER.info('Loading encoder...')
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()

    # Initialize the decoder.
    decoder = Decoder(settings.HIDDEN_SIZE, len(action_vocab))
    decoder.to(settings.DEVICE)
    settings.LOGGER.info('Loading decoder...')
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()

    # Preprocess query string.
    intent = nltk.word_tokenize(settings.QUERY.lower())

    # -----------
    # Evaluation.
    # -----------

    actions = []
    with torch.no_grad():
        intent_tensor = tensor_from_sequence(intent_vocab, intent)
        encoder_hidden = encoder.init_hidden()
        encoder_outputs = torch.zeros(
                max_action_len, settings.HIDDEN_SIZE, device=settings.DEVICE)

        for ei in range(intent_tensor.size()[0]):
            encoder_output, encoder_hidden = encoder(intent_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=settings.DEVICE)
        decoder_hidden = encoder_hidden

        for di in range(max_action_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            actions.append(action_vocab.idx2word[topi.item()])
            if topi.item() == EOS_token:
                break
            decoder_input = topi.squeeze().detach()

    # Print the solution.
    settings.LOGHANDLER.stream.write(f'Inference:\n')
    settings.LOGHANDLER.stream.write(f'> {settings.QUERY}\n')
    try:
        guess = settings.UNPARSE(actions_to_syntax_tree(actions))
        settings.LOGHANDLER.stream.write(f'< {guess}\n')
    except:
        print_actions(actions, settings.LOGHANDLER.stream)
    settings.LOGGER.info(f'Done!')

# ============================================================================
# Main entry point of SASN.
# ============================================================================

if __name__ == '__main__':

    # -----------------------------
    # Parse command line arguments.
    # -----------------------------

    def increase_subcommand_max_position(prog):
        hf = argparse.HelpFormatter(prog, max_help_position=45)
        hf._action_max_length = 14  # Fix for bpo-25297.
        return hf

    parser = argparse.ArgumentParser(
            formatter_class=increase_subcommand_max_position,
            description='SASN is a Simple Abstract Syntax Network',
            epilog='for further help run "main.py {subcommand} --help"')

    subparsers = parser.add_subparsers(
            title='subcommands', metavar='MODE', dest='MODE', required=True)

    # Arguments common to all subparsers.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('dataset',
            help='the name of the module to import settings from')
    config_parser.add_argument('--set', metavar='KEY=VALUE', action='append',
            help='keyword arguments to add/overwrite settings with')

    def increase_help_max_position(prog):
        return argparse.HelpFormatter(prog, max_help_position=45)

    # Grammargen subparser.
    grammargen_parser = subparsers.add_parser('grammargen',
            formatter_class=increase_help_max_position,
            parents=[config_parser],
            help='generate the language modules of a dataset',
            description='generate the language modules of a dataset')

    # Preprocess subparser.
    preprocess_parser = subparsers.add_parser('preprocess',
            formatter_class=increase_help_max_position,
            parents=[config_parser],
            help='turn JSON data into a preprocessed binary datafile',
            description='turn JSON data into a preprocessed binary datafile')
    preprocess_parser.add_argument('JSON', type=Path, metavar='json',
            help='the input JSON datafile')
    preprocess_parser.add_argument('PICKLE', type=Path, metavar='pickle',
            help='the output binary (pickled) datafile')

    # Train subparser.
    train_parser = subparsers.add_parser('train',
            formatter_class=increase_help_max_position,
            parents=[config_parser],
            help='train a model on a preprocessed datafile',
            description='train a model on a preprocessed datafile')
    train_parser.add_argument('DATAFILE', type=Path, metavar='datafile',
            help='the preprocessed train datafile')
    train_parser.add_argument('MODEL', type=Path, metavar='model',
            help='the destination file for the model binary')

    # Query subparser.
    query_parser = subparsers.add_parser('query',
            formatter_class=increase_help_max_position,
            parents=[config_parser],
            help='produce results from a model against manual input',
            description='produce results from a model against manual input')
    query_parser.add_argument('MODEL', type=Path, metavar='model',
            help='the model binary')
    query_parser.add_argument('QUERY', type=str, metavar='query',
            help='the query string')

    args = parser.parse_args()

    # -------------------
    # Configure settings.
    # -------------------

    settings.configure(args.dataset)

    for key, value in args._get_kwargs():
        if key.isupper():
            setattr(settings, key, value)

    if args.set:
        for setting in args.set:
            key, value = setting.split('=')
            if key.isupper():
                setattr(settings, key, value)

    # ------------------------------
    # Application-wide logger setup.
    # ------------------------------

    logger = logging.getLogger('SASN logger')
    logger.setLevel(settings.LOGLEVEL)
    settings.LOGHANDLER.setFormatter(settings.LOGFORMATTER)
    logger.addHandler(settings.LOGHANDLER)
    settings.LOGGER = logger
    settings.LOGGER.info('SASN started.')
    settings.LOGGER.info(f'SASN settings: {settings}')

    # ------------------------------
    # Deterministic execution setup.
    # ------------------------------

    random.seed(settings.SEED)
    np.random.seed(int(settings.SEED))  # No auto-conversion in Numpy.
    torch.manual_seed(settings.SEED)
    torch.cuda.manual_seed(settings.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------
    # Execution modes.
    # ----------------

    if settings.MODE == 'grammargen':
        grammargen()
    elif settings.MODE == 'preprocess':
        preprocess()
    elif settings.MODE == 'train':
        train()
    elif settings.MODE == 'query':
        query()
