import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
# from transformers import BertTokenizer, BertModel
from dataset import MIDIRepresentationDataset, collate_fn, get_file_paths
from config import DATASET_PATH

class TransformerModel(pl.LightningModule):
    def __init__(self, pitch_vocab_size, time_vocab_size, duration_vocab_size, velocity_vocab_size, instrument_vocab_size,
                 d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_length, lr):
        super(TransformerModel, self).__init__()
        self.save_hyperparameters()

        self.pitch_embedding = nn.Embedding(pitch_vocab_size, d_model)
        self.start_embedding = nn.Embedding(time_vocab_size, d_model)
        self.duration_embedding = nn.Embedding(duration_vocab_size, d_model)
        self.velocity_embedding = nn.Embedding(velocity_vocab_size, d_model)
        self.instrument_embedding = nn.Embedding(instrument_vocab_size, d_model)

        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )

        self.pitch_pred = nn.Linear(d_model, pitch_vocab_size)
        self.start_pred = nn.Linear(d_model, time_vocab_size)
        self.duration_pred = nn.Linear(d_model, duration_vocab_size)
        self.velocity_pred = nn.Linear(d_model, velocity_vocab_size)
        self.instrument_pred = nn.Linear(d_model, instrument_vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.lr = lr

        self.automatic_optimization = False

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.pitch_embedding(src[:, :, 0]) + self.start_embedding(src[:, :, 1]) + self.duration_embedding(src[:, :, 2]) + self.velocity_embedding(src[:, :, 3]) + self.instrument_embedding(src[:, :, 4]) + self.positional_encoding[:, :src.size(1), :]
        tgt_emb = self.pitch_embedding(tgt[:, :, 0]) + self.start_embedding(tgt[:, :, 1]) + self.duration_embedding(tgt[:, :, 2]) + self.velocity_embedding(tgt[:, :, 3]) + self.instrument_embedding(tgt[:, :, 4]) + self.positional_encoding[:, :tgt.size(1), :]
        # + self.positional_encoding[:, :src.size(1), :]
        # tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        output = self.transformer(
            src_emb.transpose(0, 1),
            tgt_emb.transpose(0, 1),
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return [    # pitch, start, duration, velocity, instrument
            self.pitch_pred(output),
            self.start_pred(output),
            self.duration_pred(output),
            self.velocity_pred(output),
            self.instrument_pred(output)
        ]

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'][:, :-1]
        attention_mask = batch['attention_mask'][:, :-1]
        labels = batch['labels']

        tgt_input = labels[:, 1:]
        tgt_output = [labels[:, 1:, x] for x in range(5)]


        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)

        logits = self(input_ids, tgt_input, tgt_mask=tgt_mask, src_key_padding_mask=(attention_mask == 0))
        # Compute and backpropagate loss for each head separately
        total_loss = 0
        for logit, tgt in zip(logits, tgt_output):
            loss = self.criterion(logit.view(-1, logit.size(-1)), tgt.view(-1))
            self.manual_backward(loss, retain_graph=True)
            total_loss += loss.item()

        # Average the total loss
        avg_loss = total_loss / len(logits)
        self.log('train_loss', avg_loss)
        print("Loss", avg_loss)
        return torch.tensor(avg_loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer



if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 64
    # dataset = TransformerDataset(texts, tokenizer, max_length)
    dataset = MIDIRepresentationDataset(get_file_paths(DATASET_PATH), max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    vocab_size = 12
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    lr = 1e-4

    # TODO: Get these values
    pitch_vocab_size = 12
    time_vocab_size = 9700
    duration_vocab_size = 1600
    velocity_vocab_size = 129
    instrument_vocab_size = 129

    model = TransformerModel(pitch_vocab_size, time_vocab_size, duration_vocab_size, velocity_vocab_size, instrument_vocab_size,
                             d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_length, lr)
    trainer = pl.Trainer(max_epochs=100, callbacks=[ModelCheckpoint(monitor='train_loss')])
    trainer.fit(model, dataloader)
