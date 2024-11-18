import torch, torchvision, sys

sys.path.append('..')
import common_testing

from common_testing import test_model


common_testing.batch_size = 100

NAME_JSON_FILE = 'data.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"

# batch_size = 300 # rtx5000 24gb :)

if __name__ == '__main__':
    model = torchvision.models.efficientnet_b1(progress=True, pretrained=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(0)
    test_model(model, "BasicMLP_np_50_try", num_epochs=1, progress_bar=False)
