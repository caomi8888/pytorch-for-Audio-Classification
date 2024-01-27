import torch
import torchaudio
import train_test_split as tts
from torch import nn
from torch.utils.data import DataLoader

from urbansounddata_processing import UrbanSoundDataset
from cnn import CNNNetwork
from cnn_drop import CNNNetwork_Drop

BATCH_SIZE = 512 #256 ->512
EPOCHS = 100
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "/Users/caomi/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "/Users/caomi/Downloads/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050



def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, train_data_loader, valid_data_loader, loss_fn, optimiser, device):
    for input, target,  in train_data_loader:
        input, target = input.to(device), target.to(device)


        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    correct = 0
    total = 0
    for valid_i, v_target in valid_data_loader:
        valid_i, v_target = valid_i.to(device), v_target.to(device)
        v_prediction = model(valid_i)
        _, predicted = torch.max(v_prediction, dim = 1)
        total += v_target.shape[0]
        correct += int((predicted == v_target).sum())




    print(f"train loss: {loss.item()}, valid accuracy: {correct / total}")


def train(model, train_data_loader, valid_data_loader, loss_fn, optimiser, device, epochs):

    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, train_data_loader, valid_data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_file = tts.train_csv_path
    validation_file = tts.validation_csv_path
    test_file = tts.test_csv_path


    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128 #64 ->128
    )

    usd_train = UrbanSoundDataset(train_file,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    usd_validation = UrbanSoundDataset(validation_file,
                                  AUDIO_DIR,
                                  mel_spectrogram,
                                  SAMPLE_RATE,
                                  NUM_SAMPLES,
                                  device)

    train_dataloader = create_data_loader(usd_train, BATCH_SIZE)
    valid_dataloader = create_data_loader(usd_validation, BATCH_SIZE)


    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    #cnn_drop = CNNNetwork_Drop().to(device)

    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.functional.cross_entropy
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader,valid_dataloader ,loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "cnn_no-trans.pth")
    print("Trained feed forward net saved at cnn_drop_50.pth")
