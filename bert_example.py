# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build BERT Examples from text (source, target) pairs."""

import collections
import itertools
from typing import Mapping, MutableSequence, Optional, Sequence, Tuple

import felix_constants as constants

import tokenization
import utils
import logging

logger = logging.getLogger('__log__')


class TaggingExampleBuilder:
    """Builder class for BertExample objects.

    Attributes:
      label_map: Mapping from tags to tag IDs.
      tokenizer: A tokenization.FullTokenizer, which converts between strings and
        lists of tokens.
    """
    def __init__(self,
                 label_map,
                 max_seq_length,
                 do_lower_case,
                 vocab_file=None,):
        """Initializes an instance of TaggingExampleBuilder.

        Args:
          label_map: Mapping from tags to tag IDs.
          max_seq_length: Maximum sequence length.
          do_lower_case: Whether to lower case the input text. Should be True for
            uncased models and False for cased models.
          converter: Converter from text targets to points.
          use_open_vocab: Should MASK be inserted or phrases. Currently only True is
            supported.
          vocab_file: Path to BERT vocabulary file.
          converter_insertion: Converter for building an insertion example based on
            the tagger output. Optional.
          special_glue_string_for_sources: If there are multiple sources, this
            string is used to combine them into one string. The empty string is a
            valid value. Optional.
        """
        self.labels = label_map
        self.label_ids_punc = {k:v for k, v in self.labels['punc'].items() }
        self.label_ids_cws = {k:v for k, v in self.labels['cws'].items() }
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=do_lower_case)
        self._max_seq_length = max_seq_length
        self._pad_id = self._get_pad_id()
        self._do_lower_case = do_lower_case
        
        self.cls_tag_cnt = [0, 0]
    def get_labels(self):
        return self.labels
    
    def build_example(self, source, punc_target, cws_target, input_format):
        if input_format == 'raw':
            return self.build_tagging_example(source, punc_target, cws_target)
        else:
            raise f" invalid input format {input_format}"

    def build_tagging_example(self, source, punc_target, cws_target):
        post_source = source
        if self._do_lower_case:
            post_source = post_source.lower()
    
        # 中文以字处理
        tokens = post_source.split()
        punc_tags = punc_target.split()
        cws_tags  = cws_target.split()
        
        input_tokens =   self._truncate_list(tokens)
        punc_tags = self._truncate_list(punc_tags)
        cws_tags = self._truncate_list(cws_tags)

        input_ids    = self.tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [0] * len(input_ids)
        input_mask  = [1.0] * len(input_ids)
        
        punc_tags_id = [self.label_ids_punc[t] for t in punc_tags]
        cws_tags_id = [self.label_ids_cws[t] for t in cws_tags]
        
        labels_mask = [1.0 ] * len(punc_tags_id)

        tagging_example = FeedExample(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            punc_edit_ids=punc_tags_id,
            cws_edit_ids=cws_tags_id,
            labels_mask=labels_mask,
            input_tokens=input_tokens,
            source_text=post_source,
            target_punc_text=punc_target,
            target_cws_text=cws_target)
        
        return tagging_example

    def _split_to_wordpieces(self, tokens):
        """Splits tokens to WordPieces.

        Args:
          tokens: Tokens to be split.

        Returns:
          List of WordPieces.
        """
        bert_tokens = []  # Original tokens split into wordpieces.
        special_tokens = {constants.SEP.lower(), constants.CLS.lower()}

        for token in tokens:
            # Don't tokenize special tokens.
            if token.lower() not in special_tokens:
                pieces = self.tokenizer.tokenize(token) #self.tokenizer.tokenize('a')
            else:
                pieces = [token]

            bert_tokens.extend(pieces)

        return bert_tokens

    def _truncate_list(self, x):
        return x[:self._max_seq_length - 2]

    def _get_pad_id(self):
        """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
        try:
            return self.tokenizer.convert_tokens_to_ids([constants.PAD])[0]
        except KeyError:
            return 0

def cut(text):
    source_words = text.split()
    labels = []
    slabels = []
    for word in source_words:
        wsize = len(word)
        if wsize == 1:
            tags = [0]
            stags = ['S']
        elif wsize == 2:
            tags = [1, 3]
            stags = ['B', 'E']
        elif wsize > 2:
            tags = [1] + [2] * (wsize - 2) + [3]
            stags = ['B'] + ['I'] * (wsize - 2) + ['E']
        labels.extend(tags)
        slabels.extend(stags)
    
    return source_words, labels, slabels

class FeedExample:
    def __init__(self,
                    input_ids,
                    input_mask,
                    segment_ids,
                    punc_edit_ids,
                    cws_edit_ids,
                    labels_mask,
                    input_tokens,
                    source_text,
                    target_punc_text,
                    target_cws_text):
        
        self.features = collections.OrderedDict([
            ('input_ids', input_ids),
            ('input_mask', input_mask),
            ('segment_ids', segment_ids),
            ('punc_edit_ids', punc_edit_ids),
            ('cws_edit_ids', cws_edit_ids)
        ])

        self.features_float = collections.OrderedDict([
            ('labels_mask', labels_mask),
        ])
        self.scalar_features = collections.OrderedDict()
        self.debug_features = collections.OrderedDict()
        
        self.debug_features['input_tokens'] = input_tokens
        
        if source_text is not None:
            self.debug_features['source_text'] = source_text
        if target_punc_text is not None:
            self.debug_features['target_punc_text'] = target_punc_text
        if target_cws_text is not None:
            self.debug_features['target_cws_text'] = target_cws_text
        

    def pad_to_max_length(self, max_seq_length, pad_token_id):
        """Pad the feature vectors so that they all have max_seq_length.

        Args:
          max_seq_length: The length that features will have after padding.
          pad_token_id: input_ids feature is padded with this ID, other features
            with ID 0.
        """

        for key, feature in itertools.chain(self.features.items(), self.features_float.items()):
            if key == 'cls_tag':
                continue
            pad_len = max_seq_length - len(feature)
            pad_id = pad_token_id if key == 'input_ids' else 0

            feature.extend([pad_id] * pad_len)
            if len(feature) != max_seq_length:
                raise ValueError('{} has length {} (should be {}).'.format(key, len(feature), max_seq_length))

    def to_pt_example(self):
        """Returns this object """

        pt_features = collections.OrderedDict([(key, val) for key, val in self.features.items()])
        # Add scalar integer features.
        for key, value in self.scalar_features.items():
            pt_features[key] = value # utils._int_feature(value)

        # Add label mask feature.
        for key, value in self.features_float.items():
            pt_features[key] = value  # utils._float_feature(value)

        # Add debug fields.
        for key, value in self.debug_features.items():
            pt_features[key] = value

        return pt_features
