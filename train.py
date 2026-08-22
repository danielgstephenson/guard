import torch
from value import ValueModel
import os
from typing import Any

checkpoint_path = './checkpoints/checkpoint.pt'
gen_model = ValueModel()
model = ValueModel()
opt = torch.optim.AdamW(model.parameters(),lr=1e-4)

step_count = 500
precision = 1
horizon = 0.2
batch_size = 100
batch = 0

def save_checkpoint():
    os.makedirs('./checkpoints', exist_ok=True)
    checkpoint: dict[str, Any] = {
        'model': model.state_dict(),
        'gen_model': gen_model.state_dict(),
        'horizon': horizon,
        'precision': precision,
        'opt': opt.state_dict(),
        'batch': batch
    }
    try:
        torch.save(checkpoint, checkpoint_path)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected. Saving checkpoint...')
        torch.save(checkpoint, checkpoint_path)
        print('Checkpoint saved.')
        raise

if os.path.exists(checkpoint_path):
    print(f'Loading Checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    gen_model.load_state_dict(checkpoint['gen_model'])
    horizon = checkpoint['horizon']
    precision = checkpoint['precision']
    opt.load_state_dict(checkpoint['opt'])
    batch = checkpoint['batch']
else:
    save_checkpoint()