'''
modeling disf
'''
import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel, ElectraModel,ElectraPreTrainedModel
from crf import CRFModule

class PreTrained4SequenceLabeling(BertPreTrainedModel, torch.nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.tsfm = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_rnn_size, num_layers=config.num_rnn_layers, bidirectional=True, batch_first=True)
        self.ner_classifier_tsfm = nn.Linear(config.hidden_size, config.num_labels)
        self.ner_classifier_rnn = nn.Linear(config.hidden_rnn_size * 2, config.num_labels)
        self.usecrf = config.usecrf
        self.crf = CRFModule(num_tags=config.num_labels, batch_first=True)
        self.tsfm.from_pretrained('hfl/chinese-bert-wwm-ext')
        #self.electra.from_pretrained('hfl/chinese-electra-180g-base-discriminator')
		
    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.usecrf:
            return self.forward_crf(input_ids, attention_mask, labels)
        else:
            return self.forward_tsfm(input_ids, attention_mask, labels)
        
    def forward_crf(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = torch.ne(input_ids, 0).float()
        
        discriminator_hidden_states = self.tsfm(input_ids, attention_mask)
        discriminator_sequence_output_d = self.dropout(discriminator_hidden_states[0])
        rnn_out, _ = self.lstm(discriminator_sequence_output_d, )
        
        logits = self.ner_classifier_rnn(rnn_out)
        
        if labels is not None :
            loss = -self.crf.forward_loss(logits, labels, attention_mask.bool())
            crf_labels = self.crf.forward(logits, attention_mask.bool())
            output = (loss, logits, crf_labels)
        else:
            crf_labels = self.crf.forward(logits, attention_mask.bool())
            output = (torch.tensor(0, dtype=torch.float, device=input_ids.device ), logits, crf_labels)
        
        return output
    
    def forward_tsfm(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = torch.ne(input_ids, 0).float()
        
        discriminator_hidden_states = self.tsfm(input_ids, attention_mask)
        discriminator_sequence_output = discriminator_hidden_states[0]
        
        lstm_out, _ = self.lstm(discriminator_sequence_output, )
        logits = self.ner_classifier_rnn(lstm_out)

        pred_labels = torch.argmax(logits, dim=2)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            #Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        else:
            loss = torch.tensor(0, dtype=torch.float, device=input_ids.device )
        
        return loss, pred_labels

