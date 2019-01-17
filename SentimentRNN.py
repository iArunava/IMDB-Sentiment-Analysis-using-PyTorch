import torch
import torch.nn as nn

class SentimentRNN(nn.module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers
        """
        super(SentimentRNN, self).__init__()

        # Define the class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.train_on_gpu = torch.cuda.is_available()

        # Define all layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = drop_prob,
                            batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state
        """
        batch_size = x.size(0)

        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = x.contiguos().view(-1, self.hidden_dim)

        out = self.dropout(x)
        out = self.fc(out)
        out_sig = self.sigmoid(out)

        out_sig = out.view(batch_size, -1)
        sig_out = out[:, -1]

        # Return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """
        Initializes hidden state
        """

        weights = next(self.parameters()).data

        if (self.train_on_gpu):
            hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())

        else:
            hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weights.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
