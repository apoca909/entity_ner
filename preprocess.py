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

"""Utility functions for preprocessing Felix* examples."""

import bert_example
import utils


def initialize_builder(label_map_file, max_seq_length, vocab_file, do_lower_case):
  """Returns a builder for tagging and insertion BERT examples."""

  labels = utils.read_label_map(label_map_file)
  builder = bert_example.TaggingExampleBuilder(
        label_map=labels,
        vocab_file=vocab_file,
        max_seq_length=max_seq_length,
        do_lower_case=do_lower_case
        )
  return builder
