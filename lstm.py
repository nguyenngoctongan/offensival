import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import glob
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext import data
import torch.nn as nn
import torch.optim as optim
import time
import sys
import spacy
nlp = spacy.load('en_core_web_sm')

def get_train_set(data_folder, inputfile, subtask):
    """
    Read training file
    """
    train_folder = data_folder+"_subtask"+subtask
    if not os.path.isdir(train_folder):
            os.mkdir(train_folder)
    file_name = 'train.tsv'
    path_to_file = train_folder+"/"+file_name

    df = pd.read_table(inputfile, delimiter="\t")
    train_data = pd.DataFrame()
    if subtask == "a":
        train_data['tweet'] = df["tweet"]
        train_data['label'] = df["subtask_a"].progress_map(lambda label: 1 if label == 'OFF' else 0)
        train_data.to_csv(path_to_file,sep='\t')
        return train_folder, file_name
        
    elif subtask == "b":
        filtered_data = df.loc[df['subtask_a'] == "OFF"]
        train_data['tweet'] = filtered_data["tweet"]
        train_data['label'] = filtered_data["subtask_b"].progress_map(lambda label: 1 if label == 'TIN' else 0)
        train_data.to_csv(path_to_file,sep='\t')
        return train_folder, file_name
        
    else:
        return "Warning!!!: 'subtask' argument only takes either 'a' or 'b'"

def get_test_set(data_folder,train_folder,subtask):
    tweet_file = data_folder+"/testset-level"+subtask+'.tsv'
    tweets = pd.read_table(tweet_file, delimiter = "\t")
    label_file = data_folder+"/labels-level"+subtask+'.csv'
    labels = pd.read_csv(label_file, names=['id', 'label'])
    if subtask == "a":
        labels["label"] = labels["label"].progress_map(lambda label: 1 if label == 'OFF' else 0)
    elif subtask == "b":
        labels["label"] = labels["label"].progress_map(lambda label: 1 if label == 'TIN' else 0)
    else:
        return "Warning!!!: 'subtask' argument only takes either 'a' or 'b'"
    merged_data = pd.merge(tweets, labels, on="id")
    test_data = merged_data.drop(columns="id")
    file_name = "test.tsv"
    path_to_file = train_folder +'/'+file_name
    test_data.to_csv(path_to_file,sep='\t')

    return file_name


def get_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name())
        return device
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
        return device

def save_vocab(TWEET, vocab_file):
    with open(vocab_file, 'w+', encoding="utf-8") as f:     
        for token, index in TWEET.vocab.stoi.items():
            f.write(f'{index}\t{token}')
    return vocab_file

def prepare_data(data_folder, train_file,test_file, device):
    TWEET = data.Field(sequential=True,tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',lower=True, include_lengths=True)
    LABEL = data.LabelField(dtype = torch.float)

    fields = [(None, None),("tweet", TWEET), ("label", LABEL)]

    train_set, test_set = data.TabularDataset.splits(
        path = data_folder,
        train = train_file,
        test = test_file,
        format = 'tsv',
        skip_header = True,
        fields = fields) 
    print(vars(test_set[3]))

    TWEET.build_vocab(train_set)
    MAX_VOCAB_SIZE = 25000
    TWEET.build_vocab(train_set, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.twitter.27B.100d", 
                 unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(train_set)
    save_vocab(TWEET, "vocab.txt")
    BATCH_SIZE = 64

    train_iterator, test_iterator = data.BucketIterator.splits( 
        (train_set, test_set),
        sort_key = lambda x: len(x.tweet), #sort by s attribute (quote)
        batch_size=BATCH_SIZE, device=device)

    print(vars(list(train_iterator)[0]))

    train_dataloader = list(train_iterator)
    test_dataloader = list(test_iterator)

    return train_dataloader, test_dataloader, TWEET, LABEL


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, text, text_lengths):
        
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths.to('cpu'),enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        cat = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        rel = self.relu(cat)    
        preds = self.fc(rel)
        
        return preds

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    accuracy = correct.sum() / len(correct)
    return accuracy

def train(model, iterator, loss_funct):
    
    optimizer = optim.Adam(model.parameters())
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in tqdm(iterator):
        
        optimizer.zero_grad()
        
        text, text_lengths = batch.tweet
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = loss_funct(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, loss_funct):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.tweet
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = loss_funct(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main(data_folder, subtask, lstm_model_path, device, N_EPOCHS = 4):

    data_file = data_folder + "/olid-training-v1.0.tsv"

    train_data_folder, train_file = get_train_set(data_folder, data_file, subtask)
    test_file = get_test_set(data_folder,train_data_folder, subtask)

    train_dataloader, test_dataloader, TWEET, LABEL = prepare_data(train_data_folder, train_file,test_file, device)

    INPUT_DIM = len(TWEET.vocab)
    PAD_IDX = TWEET.vocab.stoi[TWEET.pad_token]
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True 
    DROPOUT = 0.5

    model = LSTM(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)
    model.to(device)

    pretrained_embeddings = TWEET.vocab.vectors

    print(pretrained_embeddings.shape)
    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = TWEET.vocab.stoi[TWEET.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    print(model.embedding.weight.data)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    loss_funct = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss() #
    loss_funct = loss_funct.to(device)

    best_test_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()
    
        train_loss, train_acc = train(model, train_dataloader, loss_funct)
        test_loss, test_acc = evaluate(model, test_dataloader, loss_funct)
    
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), lstm_model_path)
    
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

    return lstm_model_path, TWEET

def read_vocab(path):
    vocab = dict()
    with open(path, 'r') as f:
        for line in f:
            index, token = line.split('\t')
            vocab[token] = int(index)
    return vocab

def make_predict(lstm_model_path, device):
    sentence = input("Input a tweet: ")
    model = model.load_state_dict(torch.load(lstm_model_path))
    model.eval()
    vocab = read_vocab("vocab.txt")
    tokens = [token.text for token in nlp.tokenizer(sentence)]
    indexed = [vocab.get(t, unk_idx) for t in tokens]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    
    print(prediction.item())



if __name__ == "__main__":

    data_folder = input("Data_folder: ")
    subtask = input("Subtask: ")
    lstm_model_path = 'lstm-model.pt'
    device = get_device()

    if os.path.exists(lstm_model_path):
        make_predict(lstm_model_path,device )
    else:
        main(data_folder, subtask, lstm_model_path, device)