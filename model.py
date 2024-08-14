import argparse
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
# from transformers import BertTokenizer, BertModel
from dataset import MIDIRepresentationDataset, collate_fn, get_file_paths
from config import DATASET_PATH
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
import wandb
from position_encoding import positional_encoding_classes
from datetime import datetime
from concepts import CONCEPTS

class TransformerModel(pl.LightningModule):
    def __init__(self, pitch_vocab_size, time_vocab_size, duration_vocab_size, velocity_vocab_size, instrument_vocab_size,
                 d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_length, lr, positional_encoding="base", use_concepts=True):
        super(TransformerModel, self).__init__()
        self.save_hyperparameters()

        self.pitch_embedding = nn.Embedding(pitch_vocab_size, d_model)
        self.start_embedding = nn.Embedding(time_vocab_size, d_model)
        self.duration_embedding = nn.Embedding(duration_vocab_size, d_model)
        self.velocity_embedding = nn.Embedding(velocity_vocab_size, d_model)
        self.instrument_embedding = nn.Embedding(instrument_vocab_size, d_model)

        # self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        if positional_encoding not in positional_encoding_classes:
            raise ValueError(f"positional_encoding should be one of {list(positional_encoding_classes.keys())}")
        self.positional_encoding = positional_encoding_classes[positional_encoding](d_model, max_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        self.pitch_pred = nn.Linear(d_model, pitch_vocab_size)
        self.start_pred = nn.Linear(d_model, time_vocab_size)
        self.duration_pred = nn.Linear(d_model, duration_vocab_size)
        self.velocity_pred = nn.Linear(d_model, velocity_vocab_size)
        self.instrument_pred = nn.Linear(d_model, instrument_vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.lr = lr

        self.automatic_optimization = False

        self.use_concepts = use_concepts
        if use_concepts:
            self.concept_predictors = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(len(CONCEPTS))])
            self.concept_criterion = nn.MSELoss()

    def forward(self, src, tgt,
                src_mask=None,
                tgt_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                octaves_src=None,
                octaves_tgt=None,
                absolute_start_src=None,
                absolute_start_tgt=None
                ):
        # print("Min start", torch.min(src[:, :, 1]))
        # print(src[:, :, 1])

        pos_encoding_input_src = {
            "x": src,
            "time": absolute_start_src,
            "octave": octaves_src
        }

        pos_encoding_input_tgt = {
            "x": tgt,
            "time": absolute_start_tgt,
            "octave": octaves_tgt
        }
        src_emb = self.pitch_embedding(src[:, :, 0]) + self.start_embedding(src[:, :, 1]) + self.duration_embedding(src[:, :, 2]) + self.velocity_embedding(src[:, :, 3]) + self.instrument_embedding(src[:, :, 4]) + self.positional_encoding(pos_encoding_input_src)
        tgt_emb = self.pitch_embedding(tgt[:, :, 0]) + self.start_embedding(tgt[:, :, 1]) + self.duration_embedding(tgt[:, :, 2]) + self.velocity_embedding(tgt[:, :, 3]) + self.instrument_embedding(tgt[:, :, 4]) + self.positional_encoding(pos_encoding_input_tgt)
        # + self.positional_encoding[:, :src.size(1), :]
        # tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        memory = self.encoder(
            src_emb.transpose(0, 1), 
            src_mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask
        )

        #  TODO: Add concept predictions here
        concept_predictions = []
        if self.use_concepts:
            concept_predictions = [predictor(memory) for predictor in self.concept_predictors]

        output = self.decoder(
            tgt_emb.transpose(0, 1), 
            memory,
            tgt_mask=tgt_mask, 
            memory_mask=src_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        final_output = [    # pitch, start, duration, velocity, instrument
            self.pitch_pred(output),
            self.start_pred(output),
            self.duration_pred(output),
            self.velocity_pred(output),
            self.instrument_pred(output)
        ]

        return final_output, concept_predictions
    def training_step(self, batch, batch_idx):
        return self._process_batch(batch, batch_idx, 'train')


    def _process_batch(self, batch, batch_idx, mode):
        # torch.autograd.set_detect_anomaly(True)
        opt = self.optimizers()
        input_ids = batch['input_ids'][:, :-1]
        labels = batch['labels']
        concepts = batch['concepts']

        tgt_input = labels[:, :-1]
        tgt_output = [labels[:, 1:, x] for x in range(5)]


        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)

        logits, concept_predictions = self(input_ids, tgt_input, tgt_mask=tgt_mask,
         src_key_padding_mask=(batch['attention_mask'][:, :-1] == 0).bool(), 
         tgt_key_padding_mask=(batch['attention_mask'][:, 1:] == 0).bool(),
         absolute_start_src=batch['absolute_start'][:, :-1],
         absolute_start_tgt=batch['absolute_start'][:, 1:],
         octaves_src=batch['octaves'][:, :-1],
         octaves_tgt=batch['octaves'][:, 1:]
         )
        # Compute and backpropagate loss for each head separately
        total_loss = 0

        loss_names = ['pitch', 'start', 'duration','velocity', 'instrument']
        opt.zero_grad()
        for ind, (logit, tgt) in enumerate(zip(logits, tgt_output)):
            loss = self.criterion(logit.view(-1, logit.size(-1)), tgt.reshape(-1))
            if mode == "train": self.manual_backward(loss, retain_graph=True)
            total_loss += loss.item()

            self.log(f"{loss_names[ind]}_{mode}_loss", loss.item())

        if self.use_concepts:
            total_concept_loss = 0
            for ind, concept_description in enumerate(CONCEPTS):
                concept_groundtruth = concepts[concept_description.field_name]
                concept_loss = self.concept_criterion(concept_predictions[ind].view(-1), concept_groundtruth.float().view(-1))
                if mode == "train": self.manual_backward(concept_loss, retain_graph=True)
                total_concept_loss += concept_loss.item()
                self.log(f"{concept_description.field_name}_concept_{mode}_loss", concept_loss.item())

            self.log(f"total_concept_{mode}_loss", total_concept_loss / len(CONCEPTS))
                
        if mode == "train": 
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()

        # Average the total loss
        avg_loss = total_loss / len(logits)
        self.log(f'{mode}_loss', avg_loss)
        return torch.tensor(avg_loss)

    def validation_step(self, batch, batch_idx):
        return self._process_batch(batch, batch_idx, 'val')

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add the positional_encoding argument with the short form 'pe'
    # parser.add_argument(
    #     'positional_encoding', 
    #     choices=['base', 'time', 'pitch', 'pitch_time'], 
    #     help="Specify the type of positional encoding to use.",
    #     metavar='positional_encoding',
    #     type=str
    # )
    parser.add_argument(
        '--positional-encoding',
        '-pe',
        choices=positional_encoding_classes.keys(),
        help="Specify the type of positional encoding to use (short form).",
        default="base"
    )

    parser.add_argument(
        '--max-sequence-length',
        '-maxlen',
        type=int,
        help="Maximum sequence length used for training",
        default=256
    )

    grp = parser.add_argument_group('batch_size').add_mutually_exclusive_group()
    grp.add_argument(
        '--batch-size',
        '-b',
        type=int,
        help="Training batch size",
        default=64
    )
    grp.add_argument(
        '--batch-size-auto',
        '-bauto',
        action="store_true"
    )
    # Parse the arguments
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size if not args.batch_size_auto else 4




    wandb_logger = WandbLogger(project='symbolic_music_representation', name=f"{args.positional_encoding}-{datetime.now().strftime('%d-%m@%H:%M')}")

    try:
        wandb_logger.experiment.config["batch_size"] = BATCH_SIZE
        dataset = MIDIRepresentationDataset(get_file_paths(DATASET_PATH), max_length=args.max_sequence_length, piano_only=True, wandb_logger=wandb_logger)

        # use 20% of training data for validation
        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = len(dataset) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)
        wandb_logger.experiment.config["train_dataset_instances"] = len(train_set)
        wandb_logger.experiment.config["valid_dataset_instances"] = len(valid_set)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=19)
        valid_loader = DataLoader(valid_set, batch_size=2, collate_fn=collate_fn, num_workers=19)

        vocab_size = 12
        d_model = 512
        nhead = 8
        num_encoder_layers = 6
        num_decoder_layers = 6
        dim_feedforward = 2048
        lr = 1e-7

        # TODO: Get these values
        pitch_vocab_size = 13
        time_vocab_size = 1002
        duration_vocab_size = 1007
        velocity_vocab_size = 129
        instrument_vocab_size = 129

        model = TransformerModel(pitch_vocab_size, time_vocab_size, duration_vocab_size, velocity_vocab_size, instrument_vocab_size,
                                d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, args.max_sequence_length, lr, positional_encoding=args.positional_encoding)
        # logger = TensorBoardLogger('tb_logs', name='my_model')

        # For topology and gradients
        wandb_logger.watch(model)
        trainer = pl.Trainer(max_epochs=100, callbacks=[ModelCheckpoint(monitor='train_loss')], logger=wandb_logger,
        # auto_scale_batch_size='binsearch' if args.batch_size_auto else None
        check_val_every_n_epoch=1,
        fast_dev_run=False
        )
        # TODO: Tune this but make the batch size and train loader part of the model
        # trainer.tune(model)
        trainer.fit(model, train_loader, valid_loader)
    except KeyboardInterrupt:
        wandb.finish()
