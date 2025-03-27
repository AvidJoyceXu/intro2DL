# # [markdown]
# # Speech Recognition with Neural Networks: Utterance to Phoneme Mapping
# 

# # [markdown]
# In this assignment, you'll build a sequence-to-sequence model that maps speech utterances to phoneme sequences. You'll implement a recurrent neural network architecture that processes speech feature vectors (MFCCs) and outputs corresponding phoneme sequences.
# 
# ## Key Components
# 
# - **Data Processing**: Work with Mel-Frequency Cepstral Coefficients (MFCCs) and handle variable-length sequences through padding and packing
# - **Neural Network Architecture**: Implement a model using basic CNNs for feature extraction, BiLSTMs for sequential modeling, and optional pyramidal BiLSTMs for downsampling
# - **CTC Loss**: Train your model using Connectionist Temporal Classification to handle sequence alignment challenges
# - **Decoding Strategies**: Explore both greedy search and beam search for phoneme sequence generation
# - **Evaluation**: Use Levenshtein Distance to measure the accuracy of your predicted phoneme sequences
# - **Submission**: Submit your results on kaggle for final evaluation.
# 
# This assignment will give you practical experience with sequence modeling techniques essential to speech recognition systems while exploring various architectural choices and optimization strategies.

# # [markdown]
# # Installs


# # [markdown]
# ## Imports

# #
import torch
import random
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import wandb
import torchaudio.transforms as tat
from torchaudio.models.decoder import cuda_ctc_decoder
import Levenshtein

from sklearn.metrics import accuracy_score
import gc

import glob

import zipfile
from tqdm.auto import tqdm
import os
import datetime


import warnings
warnings.filterwarnings('ignore')

mode = "EVAL" # "TRAIN" or "EVAL"

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

# # [markdown]
# # Google Drive

# # [markdown]
# # Uncomment this if you want to use Google Drive
# from google.colab import drive
# drive.mount('/content/gdrive')

# # [markdown]
# # Kaggle API Setup

# # [markdown]
# !pip install --upgrade --force-reinstall --no-deps kaggle==1.5.8
# !mkdir /root/.kaggle
# 
# with open("/root/.kaggle/kaggle.json", "w+") as f:
# 
#     f.write('{"username":"____________","key":"_________________________"}') # Put your kaggle username & key here
# 
# !chmod 600 /root/.kaggle/kaggle.json

# # [markdown]
# ### Download Data (Chill out it will take a while😀)

# # [markdown]
# !pip install --upgrade --force-reinstall --no-deps kaggle
# !kaggle competitions download -c hw-3-p-2-automatic-speech-recognition-asr-11-785 # Download Data
# !unzip -q hw-3-p-2-automatic-speech-recognition-asr-11-785.zip    # Unzip the dataset...

# # [markdown]
# # Config

# #
#writefile config.yaml

# Subset of dataset to use (1.0 == 100% of data)
# #
import yaml
with open("config.yaml") as file:
    config = yaml.safe_load(file)

print(config)

# #
BATCH_SIZE = config["batch_size"] # Define batch size from config
root = "/home/xly/11785/hw3p2/data" # Specify the directory to your root based on your environment

# # [markdown]
# # Setup Vocabulary (DO NOT MODIFY)

# #
# ARPABET PHONEME MAPPING
# DO NOT CHANGE

CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" :
     "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}


CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())

PHONEMES = CMUdict[:-2] #To be used for mapping original transcripts to integer indices
LABELS = ARPAbet[:-2] #To be used for mapping predictions to strings
MAP = {k: i for i, k in enumerate(PHONEMES)} # NOTE: map phonemes to integers

OUT_SIZE = len(PHONEMES) # Number of output classes
print("Number of Phonemes:", OUT_SIZE)

# Indexes of BLANK and SIL phonemes
BLANK_IDX=CMUdict.index('')
SIL_IDX=CMUdict.index('[SIL]')

print("Index of Blank:", BLANK_IDX)
print("Index of [SIL]:", SIL_IDX)

# # [markdown]
# ### Sample data inspection & sanity check

# #
test_mfcc = f"{root}/train-clean/mfcc/103-1240-0000.npy"
test_transcript = f"{root}/train-clean/transcript/103-1240-0000.npy"

mfcc = np.load(test_mfcc)
transcript = np.load(test_transcript)[1:-1] #Removed [SOS] and [EOS]

print("MFCC Shape:", mfcc.shape)
print("\nMFCC:\n", mfcc)
print("\nTranscript shape:", transcript.shape)

print("\nOriginal Transcript:\n", transcript)

# map the loaded transcript (from phonemes representation) to corresponding labels representation
mapped_transcript = [CMUdict_ARPAbet[k] for k in transcript]
print("\nTranscript mapped from PHONEMES representation to LABELS representation:\n", mapped_transcript)

# Mapping list of PHONEMES to list of Integer indexes
map = {k: i for i, k in enumerate(PHONEMES)}
print("\nMapping list of PHONEMES to list of Integer indexes:\n", map)



# # [markdown]
# # Dataset and Dataloader

# # [markdown]
# ### Train Data

# #
import torchaudio
class AudioDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    def __init__(self, type='train'):
        '''
        Initializes the dataset.
        '''
        assert type in ['train', 'dev']
        self.PHONEMES = PHONEMES
        self.subset = config['subset']
        self.type=type

        # Define directories for MFCC and transcript files
        self.mfcc_dir = os.path.join(root, f'{type}-clean/mfcc')
        self.transcript_dir = os.path.join(root, f'{type}-clean/transcript')

        # Get sorted lists of files
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        self.transcript_files = sorted(os.listdir(self.transcript_dir))

        # Compute size of data subset
        subset_size = int(self.subset * len(self.mfcc_files))
        self.mfcc_files = self.mfcc_files[:subset_size]
        self.transcript_files = self.transcript_files[:subset_size]

        assert(len(self.mfcc_files) == len(self.transcript_files))
        self.length = len(self.mfcc_files)

        # Pre-load and process all data
        self.mfccs = []
        self.transcripts = []
        
        for mfcc_file, transcript_file in zip(self.mfcc_files, self.transcript_files):
            # Load MFCC and transcript
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_file))
            transcript = np.load(os.path.join(self.transcript_dir, transcript_file))
            
            # Remove SOS and EOS tokens from transcript
            transcript = transcript[1:-1]
            
            # Convert to tensors
            mfcc = torch.FloatTensor(mfcc)
            # TODO: Tensor can't contain strings or non-numerical values
        
            # transcript = [CMUdict_ARPAbet[p] for p in transcript]
            transcript = torch.tensor([MAP[p] for p in transcript])
            
            # Store processed data
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        return self.mfccs[ind], self.transcripts[ind]

    def collate_fn(self, batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''
        # Split batch into features and labels
        batch_mfcc = [item[0] for item in batch] 
        batch_transcript = [item[1] for item in batch]

        # Get original lengths
        lengths_mfcc = [x.shape[0] for x in batch_mfcc]
        lengths_transcript = [x.shape[0] for x in batch_transcript]

        # Pad sequences
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)

        # Optional: Apply SpecAugment
        if self.type == "train":
            # Permute to (batch, freq, time) for time/freq masking
            batch_mfcc_pad = batch_mfcc_pad.permute(0, 2, 1)
            
            # Apply time and frequency masking
            transform = torchaudio.transforms.TimeMasking(time_mask_param=config['time_maskings'])
            batch_mfcc_pad = transform(batch_mfcc_pad)
            
            transform = torchaudio.transforms.FrequencyMasking(freq_mask_param=config['freq_maskings'])
            batch_mfcc_pad = transform(batch_mfcc_pad)
            
            # Permute back to (batch, time, freq)
            batch_mfcc_pad = batch_mfcc_pad.permute(0, 2, 1)

        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)


# # [markdown]
# ### Test Data

# #
# TODO
# Food for thought -> Do you need to apply transformations in this test dataset class?
class AudioDatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        """
        Initialize the test dataset
        Args:
            data_path: Path to test MFCC features
        """
        self.PHONEMES = PHONEMES

        # Define directories for MFCC and transcript files
        self.mfcc_dir = os.path.join(root, 'test-clean/mfcc')

        # Get sorted lists of files
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))

        self.length = len(self.mfcc_files)

        # Pre-load and process all data
        self.mfccs = []
        
        for mfcc_file in self.mfcc_files:
            # Load MFCC
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_file))
            
            # Convert to tensors
            mfcc = torch.FloatTensor(mfcc)
            
            # Store processed data
            self.mfccs.append(mfcc)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, ind):
        return self.mfccs[ind]

    def collate_fn(self, batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  NOTE: No data augmentation for test dataset
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''
        # Split batch into features and labels
        batch_mfcc = [item for item in batch]

        # Get original lengths
        lengths_mfcc = [x.shape[0] for x in batch_mfcc]

        # Pad sequences
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)

        return batch_mfcc_pad, torch.tensor(lengths_mfcc)


# #
# To free up ram
import gc
gc.collect()

# # [markdown]
# ### Create Datasets & Data loaders

# #
# Create objects for the dataset classes
train_data = AudioDataset() # TODO: Fill in the required parameters
val_data = AudioDataset(type='dev') # TODO: You can either use the same class for train data with some modifications or make a new one :)

# #
test_data = AudioDatasetTest() # TODO: Fill in the required parameters

# #
# Do NOT forget to pass in the collate function as an argument while creating the dataloader
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=train_data.collate_fn)

val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=True, collate_fn=val_data.collate_fn)

test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, collate_fn=test_data.collate_fn)

print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

# #
# sanity check
for data in train_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, ly.shape)
    break

# # [markdown]
# # Visualize Data

# # [markdown]
# Note: In the visualized graph, do not confuse padded regions of the MFCC sequences with time/freq augmentation masks. Remember all MFCCs were padded to the maximum sequence length

# #
import matplotlib.pyplot as plt

def visualize_batch(loader, title="Dataset Visualization", num_samples=2):
    """
    Visualizes MFCC features from a single batch of data.

    Args:
        loader (DataLoader): DataLoader object (for train, val, or test).
        title (str): Title of the plot.
        num_samples (int): Number of samples to visualize from the batch.
    """
    # Get one batch from the loader.
    batch = next(iter(loader))

    # The collate_fn we defined in the dataset class returns a tuple, where the first element is the padded MFCC tensor.
    mfcc_batch = batch[0] if isinstance(batch, (tuple, list)) else batch

    # Limit to a few samples.
    num_samples = min(num_samples, mfcc_batch.size(0))

    # Create subplots.
    fig, axes = plt.subplots(1, num_samples, figsize=(15,6))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        # Each MFCC tensor is of shape [time, frequency].
        # Transpose for visualization: time on x-axis, features on y-axis.
        mfcc_img = mfcc_batch[i].cpu().numpy().T
        im = axes[i].imshow(mfcc_img, origin="lower", aspect="auto", cmap="viridis")
        axes[i].set_title(f"Sample {i}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("MFCC Coefficient")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# Visualize batches of data
visualize_batch(train_loader, title="Training Dataset MFCCs (with Time & Frequency Masking)")


# # [markdown]
# # Network

# # [markdown]
# ## Basic network (Optional)
# 
# This is a basic block for understanding, you can skip this and move to pBLSTM one

# # [markdown]
# torch.cuda.empty_cache()
# 
# class Network(nn.Module):
# 
#     def __init__(self):
# 
#         super(Network, self).__init__()
# 
#         # TODO: Adding some sort of embedding layer or feature extractor might help performance.
#         # You can use CNNs as Embedding layer to extract features. Keep in mind the Input dimensions and expected dimension of Pytorch CNN.
#         # Food for thought -> What type of Conv layers can be used here?
#         #                  -> What should be the size of input channels to the first layer?
#         self.embedding = _________________________ #TODO
# 
#         # TODO : look up the documentation. You might need to pass some additional parameters.
#         self.lstm = nn.LSTM(input_size = __________________, hidden_size = 256, num_layers = 1) #TODO
# 
#         self.classification = nn.Sequential(
#             #TODO: Linear layer with in_features from the lstm module above and out_features = OUT_SIZE
#         )
# 
# 
#         self.logSoftmax =__________________ #TODO: Apply a log softmax here. Which dimension would apply it on ?
# 
#     def forward(self, x, lx):
#         #TODO
#         # The forward function takes 2 parameter inputs here. Why?
#         # Refer to the handout for hints
#         pass
# 

# # [markdown]
# ## Initialize Basic Network
# (If trying out the basic Network)

# # [markdown]
# torch.cuda.empty_cache()
# 
# model = Network().to(device)
# print(model)

# # [markdown]
# 
# ## ASR Network
# We define everything we need for the ASR model in separate classes, and put them all together in the end

# # [markdown]
# #### Permute class

# #
class Permute(torch.nn.Module):
    '''
    Used to transpose/permute the dimensions of an MFCC tensor.
    '''
    def forward(self, x):
        return x.transpose(1, 2) # NOTE: [batch, time, freq] -> [batch, freq, time]

# # [markdown]
# #### Pyramidal Bi-LSTM (pBLSTM) class

# #
class pBLSTM(torch.nn.Module):

    '''
    Pyramidal BiLSTM
    Read the write up/paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input? # NOTE: excluded the last time step or pad with 0s
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''

    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        self.blstm = torch.nn.LSTM(input_size=input_size*2, hidden_size=hidden_size, 
                                 bidirectional=True, batch_first=True)

    def forward(self, x_packed):
        # TODO: Pad Packed Sequence
        # Unpack the packed sequence
        x, x_lens = torch.nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        
        # Downsample and reshape
        # TODO: Call self.trunc_reshape() which downsamples the time steps of x and increases the feature dimensions as mentioned above
        # self.trunc_reshape will return 2 outputs. What are they? Think about what quantites are changing.

        x, x_lens = self.trunc_reshape(x, x_lens)
        
        # Pack the sequence again
        # TODO: Pack Padded Sequence
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, 
                                                         batch_first=True,
                                                         enforce_sorted=False)
            
        # Pass through BLSTM
        # TODO: Pass the sequence through bLSTM
        output, _ = self.blstm(x_packed)
        
        return output

    def trunc_reshape(self, x, x_lens):
        batch_size, timesteps, features = x.size()
        
        # Handle odd number of timesteps by truncating the last timestep
        if timesteps % 2 != 0:
            x = x[:, :-1, :]
            timesteps -= 1
            x_lens = torch.where(x_lens % 2 != 0, x_lens - 1, x_lens)
        
        # Reshape to reduce timesteps by factor of 2 and double features
        x = x.contiguous().view(batch_size, timesteps//2, features*2)
        
        # Reduce lengths by factor of 2
        x_lens = torch.div(x_lens, 2, rounding_mode='floor')
        
        return x, x_lens

# # [markdown]
# #### Util for LSTM
# 

# #
class LSTMWrapper(torch.nn.Module):
    '''
    Used to get only output of lstm, not the hidden states.
    '''
    def __init__(self, lstm):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm

    def forward(self, x):
        output, _ = self.lstm(x)
        return output

# # [markdown]
# #### Encoder class

# #
class Encoder(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size):
        super(Encoder, self).__init__()


        # TODO: You can use CNNs as Embedding layer to extract features. Keep in mind the Input dimensions and expected dimension of Pytorch CNN.
        # Food for thought -> What type of Conv layers can be used here? # NOTE: Conv1d
        #                  -> What should be the size of input channels to the first layer? # NOTE: 
        self.embedding = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )

        # TODO:
        self.BLSTMs = LSTMWrapper(
            # TODO: Look up the documentation. You might need to pass some additional parameters.
            torch.nn.LSTM(input_size=512, \
                          hidden_size=encoder_hidden_size, \
                          num_layers=4, \
                          bidirectional=True,
                          batch_first=True
            ) #TODO
          )

        self.pBLSTMs = torch.nn.Sequential( # How many pBLSTMs are required?
            # TODO: Fill this up with pBLSTMs - What should the input_size be?
            # Hint: You are downsampling timesteps by a factor of 2, upsampling features by a factor of 2 and the LSTM is bidirectional)
            # Optional: Dropout/Locked Dropout after each pBLSTM (Not needed for early submission)
            # https://github.com/salesforce/awd-lstm-lm/blob/dfd3cb0235d2caf2847a4d53e1cbd495b781b5d2/locked_dropout.py#L5
            # ...
            # ...
            pBLSTM(input_size=encoder_hidden_size*2, hidden_size=encoder_hidden_size),
            pBLSTM(input_size=encoder_hidden_size*2, hidden_size=encoder_hidden_size),
        )

    def forward(self, x, x_lens):
        # Where are x and x_lens coming from? The dataloader

        # TODO: Call the embedding layer
        x = x.permute(0, 2, 1)  # (batch, time, features) -> (batch, channels, time)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # (batch, channels, time) -> (batch, time, channels)
        # TODO: Pack Padded Sequence
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths=x_lens.cpu().to(torch.int64), batch_first=True, enforce_sorted=False
        )
        
        # TODO: Pass Sequence through the Bi-LSTM layer
        x_packed = self.BLSTMs(x_packed) # NOTE: the LSTMWrapper only returns the output, not the hidden states
        # TODO: Pass Sequence through the pyramidal Bi-LSTM layer
        x_packed = self.pBLSTMs(x_packed)
        # TODO: Pad Packed Sequence
        encoder_outputs, encoder_lens = torch.nn.utils.rnn.pad_packed_sequence(
            x_packed, batch_first=True
        )   
        # Remember the number of output(s) each function returns

        return encoder_outputs, encoder_lens

# # [markdown]
# #### Decoder class

# #
class Decoder(torch.nn.Module):

    def __init__(self, embed_size, output_size=41):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            Permute(),
            torch.nn.BatchNorm1d(2 * embed_size),
            Permute(),
            
            torch.nn.Linear(2 * embed_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(128, output_size)
        )

        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, encoder_out): # NOTE: Take in the encoder logits
        # Pass through MLP 
        out = self.mlp(encoder_out)
        
        # Apply log softmax for classification
        out = self.softmax(out)
        
        return out

# # [markdown]
# #### ASR Model Class

# #
class ASRModel(torch.nn.Module):

    def __init__(self, input_size, embed_size=192, output_size=len(PHONEMES)):
        super().__init__()

        # Initialize encoder and decoder
        self.encoder = Encoder(input_size=input_size, 
                             encoder_hidden_size=embed_size)
        
        self.decoder = Decoder(embed_size=embed_size, 
                             output_size=output_size)

    def forward(self, x, lengths_x):
        encoder_out, encoder_lens = self.encoder(x, lengths_x)
        decoder_out = self.decoder(encoder_out)

        return decoder_out, encoder_lens

# # [markdown]
# ## Initialize ASR Network

# #
model = ASRModel(
    input_size  = config['mfcc_features'],  #TODO,
    embed_size  = config['embed_size'], #TODO
    output_size = len(PHONEMES)
).to(device)

# #
summary(model, input_data=[x.to(device), lx.to(device)])

# # [markdown]
# # Training Config
# Initialize Loss Criterion, Optimizer, CTC Beam Decoder, Scheduler, Scaler (Mixed-Precision), etc

# #
# CTC Loss - uses log softmax internally so we don't need an extra log softmax layer
criterion = nn.CTCLoss(
    blank=BLANK_IDX,  # Index of blank token 
    reduction='mean', # Average loss over batch
    zero_infinity=True # Handle inf values that may occur with small batches
).to(device)

# AdamW optimizer with weight decay for regularization
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=0.01
)

# CTC Decoder for beam search decoding during training
decoder = cuda_ctc_decoder(
    tokens=LABELS,
    nbest=1,
    beam_size=config['train_beam_width']
)

# Cosine annealing scheduler with warm restarts
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer,
#     T_0=10,  # Initial restart period
#     T_mult=2, # Multiple for subsequent restart periods
#     eta_min=1e-6 # Minimum learning rate
# )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['epochs'],
    eta_min=1e-6
)

# Mixed Precision scaler for faster training
scaler = torch.cuda.amp.GradScaler()

# # [markdown]
# ### Decode Prediction

# #
def decode_prediction(output, output_lens, decoder: cuda_ctc_decoder, PHONEME_MAP = LABELS):
    '''
    Map the decode results from cuda_ctc_decoder to phoneme labels using PHONEME_MAP
    '''
    # Look at docs for CUDA_CTC_DECODER for more info on how it was used here:
    # https://pytorch.org/audio/main/tutorials/asr_inference_with_cuda_ctc_decoder_tutorial.html
    output = output.contiguous()
    output_lens = output_lens.to(torch.int32).contiguous()
    beam_results = decoder(output, output_lens.to(torch.int32)) #lengths - list of lengths

    pred_strings                    = []

    for i in range(len(beam_results)):
        # Create the prediction from the output of the cuda_ctc_decoder. Don't forget to map it using PHONEMES_MAP.
        # NOTE: PHONEMES_MAP should be the phoneme LABELS instead, as defined above as LABELS
        # Get the top prediction for each sequence in the batch
        top_beam_results = beam_results[i][0].tokens

        # TODO: Map the sequence of indices to actual phoneme LABELS and join them into a string
        # Append to predited strings list after joining
        pred_string = ''.join([PHONEME_MAP[idx] for idx in top_beam_results])
        pred_strings.append(pred_string)

    return pred_strings

def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP= LABELS): # y - sequence of integers

    dist            = 0
    batch_size      = label.shape[0]

    pred_strings    = decode_prediction(output, output_lens, decoder, PHONEME_MAP)

    for i in range(batch_size):
        # TODO: Get predicted string and label string for each element in the batch

        # Get actual length of this sequence (without padding)
        actual_len = label_lens[i]

        label_phonemes = label[i,:actual_len].cpu().numpy()
        label_string = ''.join([PHONEME_MAP[idx] for idx in label_phonemes]) # TODO
        pred_string = pred_strings[i] # TODO: Predicted string from decode_prediction

        dist += Levenshtein.distance(pred_string, label_string)

    # Average the distance over the batch
    dist /= batch_size # Think about why we are doing this
    return dist

# #
torch.cuda.empty_cache()
gc.collect()

# # [markdown]
# ## Test Implementation

# #
# test code to check shapes

model.eval()
for i, data in enumerate(val_loader, 0):
    x, y, lx, ly = data
    x, y = x.to(device), y.to(device)
    lx, ly = lx.to(device), ly.to(device)
    h, lh = model(x, lx)
    print(h.shape)
    h = torch.permute(h, (1, 0, 2))
    print(h.shape, y.shape)
    loss = criterion(h, y, lh, ly)
    print(loss)

    print(calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh.to(device), ly, decoder, LABELS))

    del x, y, lx, ly, h, lh, loss
    torch.cuda.empty_cache()

    break

# # [markdown]
# ## WandB

# #
# Use wandb? Resume Training?
USE_WANDB = config['wandb']

RESUME_LOGGING = False # Set this to true if you are resuming training from a previous run

# Create your wandb run

run_name = '{}_0327_new_scheduler'.format(config['Name'])

# If you are resuming an old run
if USE_WANDB:

    wandb.login(key="4dd2f46439865db4e3547d39c268ff46468b8ef4") #TODO

    if RESUME_LOGGING:
        run = wandb.init(
            id     = "", ### Insert specific run id here if you want to resume a previous run
            resume = "must", ### You need this to resume previous runs
            project = "hw3p2-ablations", ### Project should be created in your wandb
            settings = wandb.Settings(_service_wait=300)
        )


    else:
        run = wandb.init(
            name    = run_name, ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit  = True, ### Allows reinitalizing runs when you re-run this cell
            project = "hw3p2-ablations", ### Project should be created in your wandb account
            config  = config ### Wandb Config for your run
        )

        ### Save your model architecture as a string with str(model)
        model_arch  = str(model)
        ### Save it in a txt file
        arch_file   = open("model_arch.txt", "w")
        file_write  = arch_file.write(model_arch)
        arch_file.close()

        ### log it in your wandb run with wandb.save()
        wandb.save('model_arch.txt')

# # [markdown]
# # Training Functions

# #
# Train function
def train_model(model, train_loader, criterion, optimizer):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)
        lx, ly = lx.to(device), ly.to(device)

        with torch.cuda.amp.autocast():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close() # You need this to close the tqdm bar

    return total_loss / len(train_loader)


# Eval function
def validate_model(model, val_loader, decoder, phoneme_map= LABELS):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)
        lx, ly = lx.to(device), ly.to(device)

        with torch.inference_mode():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()
        vdist += calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh.to(device), ly, decoder, phoneme_map)

        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))

        batch_bar.update()

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss = total_loss/len(val_loader)
    val_dist = vdist/len(val_loader)
    return total_loss, val_dist


# # [markdown]
# ### Model Saving & Loading functions

# #
def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict() if scheduler is not None else {},
         metric[0]                  : metric[1],
         'epoch'                    : epoch},
         f"{path}_{device}_dist={metric[1]}_{epoch}.pth"
    )

def load_model(path, model, optimizer= None, scheduler=None, metric='valid_dist'):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch   = checkpoint['epoch']
    metric  = checkpoint[metric]

    # print("\nResuming training from epoch:", epoch)
    # print('----------------------------------------\n')
    # print("Epochs left: ", config['epochs'] - epoch)
    # print("Optimizer: \n", optimizer)
    # print("Current Schedueler T_cur:", scheduler.T_cur)

    # print("Best Val Dist:", metric)

    return [model, optimizer, scheduler, epoch, metric]

# # [markdown]
# ## Training Loop

# #
# Instantiate variables used in training loop
last_epoch_completed = 0
best_lev_dist = float("inf")

# # [markdown]
# #### Uncomment this if resuming training from model checkpoint

# #
# RESUME_TRAINING = True # Set this to true if you are resuming training from a mpdel checkpoint

# if RESUME_TRAINING:

#     checkpoint_path = ''
#     checkpoint = load_model(checkpoint_path, model, optimizer, scheduler, metric='valid_dist')

#     last_epoch_completed = checkpoint[3]
#     best_lev_dist = checkpoint[4]

# #
# Set up checkpoint directories and WanDB logging watch
checkpoint_root = os.path.join(os.getcwd(), 'checkpoints')
os.makedirs(checkpoint_root, exist_ok=True)
wandb.watch(model, log="all")

checkpoint_best_model_filename = 'best'
checkpoint_last_epoch_filename = 'last'
epoch_model_path = os.path.join(checkpoint_root, checkpoint_last_epoch_filename)
best_model_path = os.path.join(checkpoint_root, checkpoint_best_model_filename)

# WanDB log watch
if config['wandb']:
  wandb.watch(model, log="all")


# #
# Clear RAM for storage before you start training
torch.cuda.empty_cache()
gc.collect()

# # [markdown]
# #### Iterate over the number of epochs to train and evaluate your model
# 

# #
if mode == "TRAIN":
    for epoch in range(last_epoch_completed, config['epochs']):
        print("\nEpoch: {}/{}".format(epoch + 1, config['epochs']))

        # Get current learning rate
        curr_lr = optimizer.param_groups[0]['lr']

        # Train and validate
        train_loss = train_model(model, train_loader, criterion, optimizer)
        valid_loss, valid_dist = validate_model(model, val_loader, decoder, LABELS)

        # Update learning rate using scheduler
        scheduler.step()

        print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
        print("\tVal Dist {:.04f}\t Val Loss {:.04f}".format(valid_dist, valid_loss))

        if config['wandb']:
            wandb.log({
                'train_loss': train_loss,
                'valid_dist': valid_dist,
                'valid_loss': valid_loss,
                'lr': curr_lr
        })

        # # Save last epoch model

        # save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, epoch_model_path)
        # if config['wandb']:
        #     wandb.save(epoch_model_path)
        # print("Saved epoch model")

        if valid_dist <= best_lev_dist and valid_dist < 5.3: # NOTE: Only save model that is good enough
            best_lev_dist = valid_dist
            save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
            if config['wandb']:
                wandb.save(best_model_path)
            print("Saved best val model")

# Finish Wandb run
if config['wandb']:
    run.finish()


# # [markdown]
# # Generate Predictions and Submit to Kaggle

# #
#TODO: Make predictions

# Follow the steps below:
# 1. Create a new object for CUDA_CTC_DECODER with larger number of beams (why larger?)
# 2. Get prediction string by decoding the results of the beam decoder
model = ASRModel(
    input_size  = config['mfcc_features'],  #TODO,
    embed_size  = config['embed_size'], #TODO
    output_size = len(PHONEMES)
).to(device)

# path = "checkpoints/best_cuda:1_dist=4.639796401515151_136.pth"
# path = "checkpoints/best_cuda:1_dist=4.8258996212121215_103.pth"
path = "checkpoints/best_cuda:1_dist=4.66451231060606_128.pth"
# [model, optimizer, scheduler, epoch, metric]
model = load_model(path, model)[0]




test_decoder = cuda_ctc_decoder(tokens=LABELS, nbest=1, beam_size=config['test_beam_width']) # TODO

results = []

model.eval()
print("Testing")

for data in tqdm(test_loader):

    x, lx   = data
    x, lx   = x.to(device), lx.to(device)

    with torch.no_grad():
        h, lh = model(x, lx)

    prediction_string = decode_prediction(h, lh.to(device), test_decoder, LABELS) # TODO call decode_prediction

    #TODO save the output in results array.
    # Hint: The predictions of each mini-batch are a list, so you may want to extend the results list, instead of append
    results.extend(prediction_string)
    
    del x, lx, h, lh
    torch.cuda.empty_cache()

# #
if results:
    df = pd.DataFrame({
        'index': range(len(results)), 'label': results
    })

data_dir = "submission.csv"
df.to_csv(data_dir, index = False)

# #
# !kaggle competitions submit -c hw-3-p-2-automatic-speech-recognition-asr-11-785 -f /content/submission.csv -m "I made it! "


