import torch
import torch.nn as nn
# Custom chatbot model with 3 layers of LSTM
class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=3, dropout=0.3):
        super(ChatbotModel, self).__init__()
        # Embedding, encoding, decoding and flattening the params
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    #Initial weight
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    #Forward network with the embeddings
    def forward(self, input_ids, labels=None):
        with torch.autograd.set_detect_anomaly(True):
            embedded = self.dropout(self.embedding(input_ids))
            encoder_output, (hidden, cell) = self.encoder(embedded)
    
            if labels is not None:
                embedded_labels = self.dropout(self.embedding(labels))
                decoder_output, _ = self.decoder(embedded_labels, (hidden, cell))
                logits = self.fc(decoder_output)
                return logits
            else:
                batch_size = input_ids.size(0)
                decoder_input = input_ids[:, 0].unsqueeze(1).clone()  # BOS token 
                decoder_hidden = (hidden, cell)
                outputs = []
    
                for _ in range(50):  # max generation length
                    embedded_input = self.dropout(self.embedding(decoder_input))
                    decoder_output, decoder_hidden = self.decoder(embedded_input, decoder_hidden)
                    logits = self.fc(decoder_output)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    outputs.append(next_token)
                    decoder_input = next_token.detach()  # preventsgradient tracking issues
    
                predicted_ids = torch.cat(outputs, dim=1)
                return predicted_ids
