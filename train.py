import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
torch.set_default_device(device)

from value import ValueModel
import os
from typing import Any
from generate import generate

checkpoint_path = './checkpoints/checkpoint.pt'
gen_model = ValueModel()
model = ValueModel()
opt = torch.optim.AdamW(model.parameters(),lr=1e-4)

step_count = 10
batch_size = 500
batch_count = 100
epoch = 1

def save_checkpoint():
    os.makedirs('./checkpoints', exist_ok=True)
    checkpoint: dict[str, Any] = {
        'model': model.state_dict(),
        'gen_model': gen_model.state_dict(),
        'opt': opt.state_dict(),
        'epoch': epoch
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
    opt.load_state_dict(checkpoint['opt'])
    epoch = checkpoint['epoch']
else:
    save_checkpoint()

print('Training...')
for _ in range(100000000):
    horizon = min(0.01 * 1.05**epoch, 10.0)
    precision = min(0.01 * 1.05**epoch, 20.0)
    for batch in range(batch_count):
        state, value = generate(gen_model,batch_size,step_count,horizon,precision)
        opt.zero_grad()
        estimate = model(state)
        mse = ((value - estimate)**2).mean()
        mse.backward()
        opt.step()
        with torch.no_grad():
            null_estimate = value.mean()
            null_mse = ((value - null_estimate)**2).mean()
            r2 = 1 - mse/null_mse
            msg = f'epoch: {epoch}, '
            msg += f'batch: {batch+1}, '
            msg += f'horizon: {horizon:.03f}, '
            msg += f'precision: {precision:.3f}, '
            msg += f'R2: {r2:.03f}, '
            print(msg)
    epoch += 1
    gen_model.load_state_dict(model.state_dict())
    save_checkpoint()

