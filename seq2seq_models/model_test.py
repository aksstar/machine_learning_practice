from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from keras.models import Model, load_model
import sys
import ast
import json

latent_dim = 256  # Latent dimensionality of the encoding space.
file = open('model_data.json')
model_param = json.loads(file.read())

# Restore the model and construct the encoder and decoder.
model = load_model('s2smodel.h5')

encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in model_param['input_token_index'].items())
reverse_target_char_index = dict(
    (i, char) for char, i in model_param['target_token_index'].items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, model_param['num_decoder_tokens']))
    # Populate the first character of target sequence with the start character.
    target_token_index = model_param['target_token_index']
    target_seq[0, 0, target_token_index['<eos>']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > model_param['max_decoder_seq_length']):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, model_param['num_decoder_tokens']))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


encoder_input_data = np.zeros(
    (1, model_param['max_encoder_seq_length'], model_param['num_encoder_tokens']),
    dtype='float32')

input_token_index = model_param['input_token_index']

# Take one sequence (part of the training set)
# for trying out decoding.
# input_seq = encoder_input_data[seq_index: seq_index + 1]

import pandas as pd

file = open('movies.dat')

r_cols = ['movie_id', 'movie_name', 'movie_genre']
metadata = pd.read_csv('movies.dat', sep='::', names=r_cols,
                      encoding='latin-1', index_col='movie_id')

#input_list = [2858, 2355, 909, 1294, 3363]
input_list = [260, 110, 3512, 3930,2953]
print("Input is: ", )

print(metadata.loc[input_list])


for t, char in enumerate(input_list):
    encoder_input_data[0, t, input_token_index[str(char)]] = 1.

input_seq = encoder_input_data
decoded_sentence = decode_sequence(input_seq)
print('-')


output_list = list(filter(('<eos>').__ne__, decoded_sentence))

new_list = []
for item in output_list:
    if int(item) in input_list or int(item) in new_list:
        continue
    new_list.append(int(item))

print("Output is: ", )
print(metadata.loc[new_list])