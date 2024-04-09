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
        self.ner_classifier = nn.Linear(config.hidden_rnn_size * 2, config.ner_num_labels)
        #self.target_weight = config.weight
        self.usecrf = config.usecrf
        self.crf = CRFModule(num_tags=config.num_labels, batch_first=True)
        #self.electra.from_pretrained('hfl/chinese-electra-180g-base-discriminator')

    def forward(self, input_ids, attention_mask=None, punc_labels=None, cws_labels=None):
        if attention_mask is None:
            attention_mask = torch.ne(input_ids, 0).float()
        device = input_ids.device 
        discriminator_hidden_states = self.electra(input_ids, attention_mask)
        discriminator_sequence_output = discriminator_hidden_states[0]

        # discriminator_sequence_output_d = self.dropout(discriminator_sequence_output)
        # lstm_out, _ = self.lstm(discriminator_sequence_output_d, )
        # lstm_out_d = self.dropout(lstm_out)

        lstm_out_d = discriminator_sequence_output   #omit the lstm
        ner_logits = self.ner_classifier(lstm_out_d)
        
        cws_ids  = torch.argmax(ner_logits, dim=2)

        if punc_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            #Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_punc = ner_logits.view(-1, self.config.punc_num_labels)[active_loss]
                active_labels_punc = punc_labels.view(-1)[active_loss]
                loss_punc = loss_fct(active_logits_punc, active_labels_punc)
            else:
                loss_punc = loss_fct(ner_logits.view(-1, self.config.punc_num_labels), punc_labels.view(-1))

        else:
            loss_punc = torch.tensor(0, dtype=torch.float, device=device)

        
        output = (loss, ner_logits, punc_idx, cws_logits, cws_ids)
        return output  # output = (loss, punc_logits, punc_idx, cws_logits, cws_ids)

