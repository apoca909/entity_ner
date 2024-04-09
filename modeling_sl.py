'''
modeling disf
'''
import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel
from CRF import CRFModule

class PreTrained4SequenceLabeling(BertPreTrainedModel, torch.nn.Module):
    def __init__(self, config):
        super().__init__(config)

        self.tsfm = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_rnn_size, num_layers=config.num_rnn_layers, bidirectional=True, batch_first=True)
        self.ner_classifier = nn.Linear(config.hidden_rnn_size * 2, config.num_labels)
        #self.target_weight = config.weight
        self.usecrf = config.usecrf
        self.crf = CRFModule(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None, do_eval=None):
        if attention_mask is None:
            attention_mask = torch.ne(input_ids, 0).float()
        device = input_ids.device
        discriminator_hidden_states = self.tsfm(input_ids, attention_mask)
        discriminator_sequence_output_d = self.dropout(discriminator_hidden_states[0])
        lstm_out, _ = self.lstm(discriminator_sequence_output_d, )
        
        logits = self.ner_classifier(lstm_out)
        loss = -self.crf.forward_loss(logits, labels, attention_mask.bool())
        
        if not do_eval :
            loss = -self.crf.forward_loss(logits, labels, attention_mask.bool())
            output = (loss, logits)
            
        else:
            crf_logits = self.crf.forward(logits, attention_mask.bool())
            output = (0, crf_logits)
            
        
        return output

