import torch
from pathlib import Path


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith('.pt') or model_name.endswith('.pth'), "model_name should end with '.pt' or '.pth'"

    model_state_path = target_dir_path / model_name

    print('f[INFO] Saving Model to: {model_state_path}')
    torch.save(model.state_dict(), model_state_path)
