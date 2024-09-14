import torch
import wandb
import argparse
import model
from dataset import MIDIRepresentationDataset, collate_fn, get_file_paths
from config import DATASET_PATH
import pdb


def load_checkpoint_from_wandb(run_id, artifact_name, checkpoint_file):
    """
    Load a model checkpoint from Weights & Biases.

    Args:
        run_id (str): The Weights & Biases run ID where the checkpoint is stored.
        artifact_name (str): The name of the artifact containing the model checkpoint.
        checkpoint_file (str): The filename of the checkpoint inside the artifact.

    Returns:
        checkpoint_path (str): Path to the downloaded checkpoint file.
    """
    # Initialize a new W&B run to download artifacts
    api = wandb.Api()
    runs = api.runs("rh-iu/symbolic_music_representation", order="-created_at")
    latest_run = runs[1]
    artifact = latest_run.logged_artifacts()[0]
    artifact_dir = artifact.download()
    checkpoint_path = f'{artifact_dir}/model.ckpt'
    
    return checkpoint_path

def main():
    parser = argparse.ArgumentParser(description="Generate a song using TransformerModel")
    parser.add_argument('--run_id', type=str, required=True, help='W&B run ID where the checkpoint is stored')
    parser.add_argument('--artifact_name', type=str, required=True, help='Name of the artifact containing the model checkpoint')
    parser.add_argument('--checkpoint_file', type=str, required=True, help='Filename of the model checkpoint inside the artifact')
    parser.add_argument('--max_gen_length', type=int, default=512, help='Maximum length of the generated song')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling diversity')
    parser.add_argument('--conditioning_sequence', type=str, required=True, help='Path to the conditioning sequence file (torch.Tensor)')
    
    args = parser.parse_args()

    # Load the model
    model_checkpoint = load_checkpoint_from_wandb(args.run_id, args.artifact_name, args.checkpoint_file)

    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    lr = 1e-3

    # TODO: Get these values
    pitch_vocab_size = 13
    time_vocab_size = 1002
    duration_vocab_size = 1007
    velocity_vocab_size = 129
    instrument_vocab_size = 129
    max_sequence_length = 1024

    generator = model.TransformerModel(pitch_vocab_size, time_vocab_size, duration_vocab_size, velocity_vocab_size, instrument_vocab_size,
                            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_sequence_length, lr,
                            positional_encoding="base", use_concepts=True)

    # Assuming you have the same hyperparameters saved in the checkpoint or elsewhere

    # Load the checkpoint
    checkpoint = torch.load(model_checkpoint)
    generator.load_state_dict(checkpoint['state_dict'])

    dataset = MIDIRepresentationDataset(
            get_file_paths(DATASET_PATH),
            max_length=512,
            piano_only=True,
            wandb_logger=None,
            use_concepts=True
        )

    # Load the conditioning sequence
    conditioning_sequence = dataset[0]['input_ids'].unsqueeze(0)

    # Generate a song
    generated_song = generator.generate_song(conditioning_sequence=conditioning_sequence, max_gen_length=args.max_gen_length, temperature=args.temperature)
    
    print(generated_song)
    print(generated_song.shape)

    print()
    # Save the generated song
    torch.save(generated_song, "generated_song.pt")
    print("Generated song saved as 'generated_song.pt'")

if __name__ == "__main__":
    main()
