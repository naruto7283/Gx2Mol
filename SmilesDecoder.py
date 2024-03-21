import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from rdkit import Chem
from GeneVAE import GeneVAE
from utils import get_device
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================================
# Define the SMILES dataset
class Smiles_Dataset(Dataset):
    
    def __init__(
        self, 
        gene_expression_file, 
        cell_name,
        tokenizer,
        gene_num,
        variant
    ):
        """
        gene_expression_file: original gene data file
        cell_name: cell name, e.g., MCF7
        tokenizer: vocabulary to encode and decode SMILES
        gene_num: number of gene columns
        variant: True → Apply variant SMILES
        """
        data = pd.read_csv(
            gene_expression_file + cell_name + '.csv', 
            sep=',', 
            names=['inchikey','smiles']+['gene'+str(i) for i in range(1, gene_num+1)]
        )
        # Drop the nan row
        data = data.dropna(how='any')
        # Normalize data per gene
        #data.iloc[:, 2:] = (data.iloc[:, 2:] - data.iloc[:, 2:].mean())/data.iloc[:, 2:].std()
        
        if variant:
            # Variant SMILES
            data['smiles'] = data['smiles'].apply(self.variant_smiles)

        self.data = data

        self.tokenizer = tokenizer
    
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, index):

        smi = self.data.iloc[index]['smiles']
        # Encode SMILES strings
        encoded_smi = self.tokenizer.encode(smi)
        gene = self.data.iloc[index]['gene1':].values.astype('float32')
        
        return encoded_smi, gene

    def variant_smiles(self, smi):
        
        mol = Chem.MolFromSmiles(smi)
        atom_idxs = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_idxs)
        mol = Chem.RenumberAtoms(mol,atom_idxs)

        return Chem.MolToSmiles(mol, canonical=False)

# ============================================================================
# Define the SMILES dataLoader
class Smiles_DataLoader(DataLoader):
    
    def __init__(
        self,
        gene_expression_file,
        cell_name,
        tokenizer,
        gene_num,
        batch_size,
        train_rate=0.9,
        variant=False
    ):
        """
        gene_expression_file: original gene data file
        cell_name: cell name, e.g., MCF7
        tokenizer: vocabulary to encode and decode SMILES
        gene_num: number of gene columns
        batch_size: batch size of gene data
        train_rate: split training and testing gene data by train rate
        variant: If true, apply variant SMILES
        """

        self.gene_expression_file = gene_expression_file
        self.cell_name = cell_name
        self.tokenizer = tokenizer
        self.gene_num = gene_num
        self.batch_size = batch_size
        self.train_rate = train_rate
        self.variant = variant
        
    def collate_fn(self, batch):

        # Batch is a list of zipped encoded smiles and genes
        smiles, genes = zip(*batch)
        smi_tensors = [torch.tensor(smi) for smi in smiles]
        gene_tensors = torch.tensor(np.array(genes))
        # Pad the different lengths of tensors to the maximum length
        smi_tensors = torch.nn.utils.rnn.pad_sequence(smi_tensors, batch_first=True) # [batch_size, max_len]
        
        return smi_tensors, gene_tensors
        
    def get_dataloader(self):

        # Load dataset
        dataset = Smiles_Dataset(
            self.gene_expression_file, 
            self.cell_name,
            self.tokenizer, 
            self.gene_num,
            self.variant
        )

        train_size = int(len(dataset) * self.train_rate)
        test_size = len(dataset) - train_size

        # Split train and test data
        train_data, test_data = random_split(
            dataset=dataset, 
            lengths=[train_size, test_size], 
            generator=torch.Generator().manual_seed(0)
        )

        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=1
        )

        test_dataloader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=1
        )

        return train_dataloader, test_dataloader

# ============================================================================
# Define SmilesDecoder
class SmilesDecoder(nn.Module):

    def __init__(
        self, 
        tokenizer, 
        emb_size,
        hidden_size,
        gene_latent_size,
        num_layers, 
        dropout
    ):
        """
        tokenizer: contains SMILES string tokens
        latent_size: the latent size of GeneVAE
        emb_size: embedding dimension
        hidden_size: number of hidden neurons of RNN
        gene_latent_size: the dimension of VAE latent vector
        num_layers: number of layers
        dropout: dropout probability
        """

        super(SmilesDecoder, self).__init__()

        self.tokenizer = tokenizer
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.gene_latent_size = gene_latent_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Padding idx is required
        self.embedding = nn.Embedding(
            self.tokenizer.vocab_size, 
            self.emb_size, 
            padding_idx=self.tokenizer.char_to_int[tokenizer.pad]
        )
        # Connect the context of the latent vector of GeneVAE and embbeding 
        self.rnn = nn.LSTM(
            input_size=self.emb_size + self.gene_latent_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.tokenizer.vocab_size)
        # dim=1: Softmax on each row, and the sum of each row is 1
        self.log_softmax = nn.LogSoftmax(dim=1)
        # Initialize parameters
        self. init_params()

    def forward(self, decoder_inputs, latent_vectors):
        """
        decoder_inputs: [batch_size, max_len] -> [batch_size, max_len, emb_size]
        latent_vectors: latent features of GeneVAE [batch_size, latent_size]
        returns:
            pred: [batch_size * max_len, vocab_size]
        """
        self.rnn.flatten_parameters()

        h0, c0 = self.init_hidden(decoder_inputs.size(0))
        embeded = self.embedding(decoder_inputs)  # [batch_size, max_len, emb_size]
        # Duplicate the latent_vectors [batch_size, latent_size]
        context = latent_vectors.repeat(embeded.shape[1], 1, 1).permute(1, 0, 2) # [batch_size, max_len, latent_size]
        output, _ = self.rnn(torch.cat((embeded, context), -1), (h0, c0)) # [batch_size, max_len, hidden_size]
        logits = self.fc(output.contiguous().view(-1, self.hidden_size)) # [batch_size * max_len, vocab_size]
        pred = self.log_softmax(logits) # [batch_size * max_len, vocab_size]

        return pred

    def step(
        self, 
        decoder_input, 
        latent_vector,
        h, 
        c
    ):
        """
        Compute the output per time step

        decoder_input: [batch_size, 1]
        latent_vector: [batch_size, latent_size]
        h: hidden state [num_layers, batch_size, hidden_size]
        c: cell state [num_layers, batch_size, hidden_size]

        returns:
            pred: [batch_size, vocab_size]
            h: [batch_size, hidden_size]
            c: [batch_size, hidden_size]
        """
        self.rnn.flatten_parameters()

        embeded = self.embedding(decoder_input) # [batch_size, 1, emb_size]
        latent_vector = torch.unsqueeze(latent_vector, 1) # [batch_size, 1, latent_size]
        rnn_input = torch.cat((embeded, latent_vector), -1)
        output, (h, c) = self.rnn(rnn_input, (h, c))
        logits = self.fc(output.contiguous().view(-1, self.hidden_size))
        pred = self.log_softmax(logits) # [batch_size, vocab_size]

        return pred, h, c

    def init_hidden(self, batch_size):
        
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(get_device())
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(get_device())

        return h, c

    def init_params(self):

        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def sample(
        self, 
        max_len,
        latent_vectors
    ):
        """
        sample a batch of samples: [batch_size, max_len]
        max_len: maximum length of SMILES strings
        latent_vectors: gene expression profile features [batch_size, latent_size]
        """     
        samples = []
        batch_size = latent_vectors.size(0)

        # Initilize the hidden state and cell as zero matrix
        h, c = self.init_hidden(batch_size)
        x = torch.ones(batch_size, 1, dtype = torch.int64) * self.tokenizer.char_to_int[self.tokenizer.start]
        x = x.to(get_device())
        
        for _ in range(max_len):

            output, h, c = self.step(x, latent_vectors, h, c) # output: [batch_size, vocab_size]
            x = torch.multinomial(torch.exp(output), 1) # [batch_size, 1]
            #x = torch.max(torch.exp(output), 1)[1].unsqueeze(-1)
            samples.append(x)

        pred_smiles = torch.cat(samples, dim=1)

        return pred_smiles

    def load_model(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

        

    









        
    
    
































