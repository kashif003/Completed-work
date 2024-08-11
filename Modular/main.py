import torch
from torch import nn
from torchvision import transforms

import data_setup
import engine
import save_model


class Model(nn.Module):  # Build a model here

    '''Build a model here'''


# Define the hýperparameters here.
NUM_EPOCHS = 100
BATCH_SIZE = 32
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001

# set the path for training data and testing data here
train_dir = ''
test_dir = ''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=data_transform,
                                                                               batch_size=BATCH_SIZE)
model = Model().to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)
# select the target dir for saving model here and select the name of the model here
model_target_dir = ''
model_name = ''

save_model.save_model(model=model,
                      target_dir=model_target_dir,
                      model_name=model_name)
