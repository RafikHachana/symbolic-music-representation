from datetime import datetime
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project='symbolic_music_representation', name=f"base-{datetime.now().strftime('%d-%m@%H:%M')}")
