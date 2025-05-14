'''
copyright Alex Whelan 2025
code for running the clothing model
'''
### TO-DO ###
# adv loss
# bigger batch size
# bigger network?
# skip connections
# stretch goal: x-attn
###

from tqdm import tqdm


import numpy as np
import torch
import lpips

from model import ClothingModel
from data import VITONDataLoader, VITONDataset, save_image


def run():
  ## device
  device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
  print(f"Using device {device}")
  ## Training parameters
  CHAN_IN = 3
  CHAN_OUT = 3
  EPOCHS = 100
  BATCH_SIZE = 4
  SAVE_FREQUENCY = 100
  OUTDIR = "experiment_1"

  ## train_dataset
  dataset = VITONDataset()
  loader = VITONDataLoader(dataset, BATCH_SIZE)

  ## training loop
  model = ClothingModel(CHAN_IN, CHAN_OUT)
  model.to(device)
  optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
  l1_loss = torch.nn.L1Loss(reduction='mean')
  lpips_loss = lpips.LPIPS(net='alex').to(device) # best forward scores
  
  print("[*] Start Training")
  step_counter = 0
  progress_bar = tqdm(loader.data_loader)

  for epoch in range(EPOCHS):
    progress_bar = tqdm(loader.data_loader)

    for data in progress_bar:
      ref = torch.as_tensor(data[0], dtype=torch.float).permute(0,3,1,2)
      clothing = torch.as_tensor(data[1], dtype=torch.float).permute(0,3,1,2)
      gt = torch.as_tensor(data[2], dtype=torch.float).permute(0,3,1,2)

      ref = ref.to(device)
      clothing = clothing.to(device)
      gt = gt.to(device)

      pred = model(ref, clothing)
      pixel_loss = l1_loss(pred, gt)
      percep_loss = lpips_loss(pred, gt).mean()
      total_loss = pixel_loss + percep_loss
      total_loss.backward()
      optimiser.step()

      step_counter += 1
      progress_bar.set_description(f"Epoch {epoch}, step {step_counter}: l1 = {pixel_loss.item()}, lpips = {percep_loss.item()}")

      if step_counter % SAVE_FREQUENCY == 0:
        save_image(pred, OUTDIR, step_counter)
        torch.save(model.state_dict(), f"{OUTDIR}/latest_model.pt")


if __name__ == "__main__":
  run()
