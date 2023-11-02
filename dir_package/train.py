# train


from nn_model import *

from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks import EarlyStopping
# from pytorch_lightning.callbacks import RichProgressBar


L.seed_everything(1)

#dm - data_module
dm = LightningDataModule()
dm.prepare_data()
dm.setup()

model = LightningModule(lr=0.001)

wandb_logger = WandbLogger(log_model="all", project="proj_pytorch", name="magic-dragon-1")

# callbacks = [
#     ModelCheckpoint(
#         dirpath = "checkpoints",
#         every_n_train_steps=100,
#     ),
# ]


trainer = L.Trainer(
    max_epochs=1,
    accelerator="auto",
    devices="auto",
    logger=wandb_logger,
    # callbacks=callbacks,
    # callbacks=[RichProgressBar()],
    # callbacks=EarlyStopping('val_loss', patience=7),
)

trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

trainer.test(model, dm.test_dataloader())