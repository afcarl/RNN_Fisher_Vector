import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import cPickle

from utils import shared, set_values, get_name
from nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, forward
from optimization import Optimization


class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, models_path=None, model_path=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        if model_path is None:
            assert parameters and models_path
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = get_name(parameters)
            # Model location
            model_path = os.path.join(models_path, self.name)
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                self.parameters = cPickle.dump(parameters, f)
        else:
            assert parameters is None and models_path is None
            # Model location
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = cPickle.load(f)
            self.reload_mappings()
        self.components = {}

    def save_mappings(self, id_to_word, id_to_char, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_char': self.id_to_char,
                'id_to_tag': self.id_to_tag,
            }
            cPickle.dump(mappings, f)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self):
        """
        Load components values from disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])

    def build(self,
              dropout,
              char_dim,
              char_lstm_dim,
              char_bidirect,
              word_dim,
              word_lstm_dim,
              word_bidirect,
              lr_method,
              pre_emb,
              crf,
              cap_dim,
              training=True,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)

        # Number of capitalization features
        if cap_dim:
            n_cap = 4

        # Network variables
        is_train = T.iscalar('is_train')
        word_ids = T.ivector(name='word_ids')
        char_for_ids = T.imatrix(name='char_for_ids')
        char_rev_ids = T.imatrix(name='char_rev_ids')
        char_pos_ids = T.ivector(name='char_pos_ids')
        tag_ids = T.ivector(name='tag_ids')
       
        cap_ids = T.ivector(name='cap_ids')

        # Sentence length
      
        # Final input (all word features)
        input_dim = 0
        inputs = []
        s_len = (char_pos_ids).shape[0]
        #
        #
        # Chars inputs
        #
    
        input_dim += (char_lstm_dim * 2)
        char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')

        char_lstm_for = LSTM(char_dim, char_lstm_dim, with_batch=False,
                             name='char_lstm_for')
        char_lstm_rev = LSTM(char_dim, char_lstm_dim, with_batch=False,
                             name='char_lstm_rev')

        char_lstm_for.link(char_layer.link(word_ids))
        char_lstm_rev.link(char_layer.link(cap_ids))
        
        
        final_layer = HiddenLayer(char_lstm_dim, n_chars, name='final_char_layer',
                              activation=('softmax'))
        chars_final = final_layer.link(char_lstm_for.h)
        
        final_rev_layer = HiddenLayer(char_lstm_dim, n_chars, name='final_char_rev_layer',
                              activation=('softmax'))
        chars_rev_final = final_layer.link(char_lstm_rev.h)
        
    
        cost_chars = T.nnet.categorical_crossentropy(chars_final, char_pos_ids).mean()
        cost_chars_rev = T.nnet.categorical_crossentropy(chars_rev_final, tag_ids).mean()
        
        # Network parameters
        params = []
        if char_dim:
            self.add_component(char_layer)
            self.add_component(char_lstm_for)
            params.extend(char_layer.params)
            params.extend(char_lstm_for.params)
            self.add_component(char_lstm_rev)
            params.extend(char_lstm_rev.params)
        

        # Prepare train and eval inputs
        eval_inputs = []
        if word_dim:
            eval_inputs.append(word_ids)
        if char_dim:
            eval_inputs.append(char_for_ids)
            if char_bidirect:
                eval_inputs.append(char_rev_ids)
            eval_inputs.append(char_pos_ids)
        #if cap_dim:
        
        eval_inputs.append(tag_ids)
        eval_inputs.append(cap_ids)
        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Fetch gradients from both char_lstms
        gradients = T.grad(cost_chars, char_lstm_for.params)
        gradients_rev = T.grad(cost_chars_rev, char_lstm_rev.params)
        
        # Return forward char_lstm grads        
        f_eval = theano.function(
            inputs=eval_inputs,
            outputs=gradients,
            givens=({is_train: np.cast['int32'](0)} if dropout else {}), on_unused_input='ignore'
        )
        
	# Return reverse char_lstm grads
        f_eval_rev = theano.function(
            inputs=eval_inputs,
            outputs=gradients_rev,
            givens=({is_train: np.cast['int32'](0)} if dropout else {}), on_unused_input='ignore'
        )

        return f_eval, f_eval_rev
