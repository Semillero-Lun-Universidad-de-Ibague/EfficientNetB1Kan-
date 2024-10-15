import torch, sys

sys.path.append('..')
import common_testing
from common_testing import test_model
from models.efficientnet_kan import EfficientNetB1_KAN

NAME_JSON_FILE = 'data.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"

common_testing.batch_size = 210

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(0)

    params = {
        'grid_size': 7,
        'spline_order': 3,
        'scale_noise': 0.97,
        'scale_base': 0.79,
        'scale_spline': 0.99
    }
    model = EfficientNetB1_KAN(4, params)
    test_model(model, f"Effective_b1_KAN_5epoch_7grid", num_epochs=5, progress_bar=False)

    exit()
    params = {
        'grid_size': 64,
        'spline_order': 3,
        'scale_noise': 0.97,
        'scale_base': 0.79,
        'scale_spline': 0.99
    }
    model = EfficientNetB1_KAN(4, params)
    test_model(model, f"Effective_b1_KAN_5epoch_64", num_epochs=30, progress_bar=False)
