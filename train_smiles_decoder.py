import math
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from rdkit import Chem
from rdkit import rdBase

from GeneVAE import GeneVAE
from utils import Tokenizer, get_device, mean_similarity
from SmilesDecoder import Smiles_DataLoader, SmilesDecoder
rdBase.DisableLog('rdApp.error')

# ============================================================================
# Load data
def load_smiles_data(tokenizer, args):

    # Load smiles and gene values
    smiles_loader = Smiles_DataLoader(
        args.gene_expression_file, 
        args.cell_name,
        tokenizer,
        args.gene_num,
        batch_size=args.gene_batch_size,
        train_rate=args.train_rate,
        variant=args.variant
    )
    train_dataloader, valid_dataloader = smiles_loader.get_dataloader()

    return train_dataloader, valid_dataloader

# ============================================================================
# Generate Smiles using learned gene representations
def train_smiles_decoder(
    trained_gene_vae,
    train_dataloader, 
    valid_dataloader,
    tokenizer,
    args
):
    """
    trained_gene_vae: the pretrained GeneVAE model for gene feature extraction
    train_dataloader: splited training data (encoded smiles, genes)
    tokenizer: SMILES vocabulary
    """
    # Define SmilesDecoder
    smiles_decoder = SmilesDecoder(
        tokenizer,
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
        gene_latent_size=args.gene_latent_size,
        num_layers=args.num_layers,
        dropout=args.smiles_dropout
    ).to(get_device())

    # Criterion
    nll_loss = nn.NLLLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        params=smiles_decoder.parameters(), 
        lr=args.smiles_lr
    )
    
    # Prepare file to save results
    with open(args.smiles_decoder_train_results, 'a+') as wf:
        wf.truncate(0)
        wf.write('{},{},{},{},{},{},{},{}\n'.format(\
            'Epoch', 
            'Loss', 
            'Total', 
            'Valid', 
            'Valid_rate', 
            'Unique', 
            'Unique_rate', 
            'Similarity'
        ))

    print('\n')
    print('Training Information:')

    for epoch in range(args.smiles_epochs):

        smiles_decoder.train()
        # Train SmilesDecoder
        train_loss = []
        
        # Operate on a batch of data
        for _, (smiles, genes) in enumerate(train_dataloader):

            smiles, genes = smiles.to(get_device()), genes.to(get_device())
            # Offset one bit as the input of the decoder
            decoder_inputs = smiles[:, :-1] # [batch_size, max_len-1]

            # Extract gene expression features
            trained_gene_vae.eval()
            latent_vectors, _ = trained_gene_vae(genes) # [batch_size, latent_size]

            pred = smiles_decoder(decoder_inputs, latent_vectors) # pred: [batch_size * max_len, vocab_size]
            loss = nll_loss(pred, smiles[:, 1:].contiguous().view(-1))
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        # Evaluate valid and unique SMILES
        smiles_decoder.eval()
        valid_smiles = []
        label_smiles = []
        total_num_data = len(valid_dataloader.dataset)

        for _, (smiles, genes) in enumerate(valid_dataloader):

            smiles, genes = smiles.to(get_device()), genes.to(get_device())
            trained_gene_vae.eval()
            latent_vectors, _ = trained_gene_vae(genes)
            dec_sampled_char = smiles_decoder.sample(args.max_len, latent_vectors)
            output_smiles = ["".join(tokenizer.decode(\
                dec_sampled_char[i].squeeze().detach().cpu().numpy()
                )).strip("^$ ") for i in range(dec_sampled_char.size(0))]

            #valid_smiles.extend([smi for smi in output_smiles if Chem.MolFromSmiles(smi) and Chem.MolFromSmiles(smi)!=])
            for i in range(len(output_smiles)):

                mol = Chem.MolFromSmiles(output_smiles[i])
                if mol != None and mol.GetNumAtoms() > 1 and Chem.MolToSmiles(mol) != ' ':
                    valid_smiles.extend([output_smiles[i]])
                    label_smiles.extend(["".join(tokenizer.decode(smiles[i].squeeze().detach().cpu().numpy())).strip("^$ ")])

        unique_smiles = list(set(valid_smiles))
        valid_csv = pd.DataFrame(valid_smiles).to_csv(args.valid_smiles_file, index=False)
        
        mean_loss = np.mean(train_loss)
        valid_num = len(valid_smiles)
        valid_rate = 100*len(valid_smiles)/total_num_data
        unique_num = len(unique_smiles)

        if valid_num != 0:
            unique_rate = 100*unique_num/valid_num
            mean_sim = mean_similarity(valid_smiles, label_smiles)
        else:
            unique_rate = 100*unique_num/(valid_num+1)
            mean_sim = 0
        
        print('Epoch: {:d} / {:d}, loss: {:.3f}, Total: {:d}, valid: {:d} ({:.2f}), unique: {:d} ({:.2f}), Similarity: {:.3f}'.format(\
            epoch+1, 
            args.smiles_epochs, 
            mean_loss, 
            total_num_data, 
            valid_num, 
            valid_rate,
            unique_num,
            unique_rate,
            mean_sim
        ))
        
        # Save trained results to file
        with open(args.smiles_decoder_train_results, 'a+') as wf:
            wf.write('{},{:.3f},{},{},{:.2f},{},{:.2f},{:.3f}\n'.format(\
                epoch+1, 
                mean_loss, 
                total_num_data, 
                valid_num, 
                valid_rate,
                unique_num,
                unique_rate,
                mean_sim
            ))

        # Save predicted and label SMILES into file
        final_smiles = {'predict': valid_smiles, 'label': label_smiles}
        final_smiles = pd.DataFrame(final_smiles)
        # Save to file
        final_smiles.to_csv(args.valid_smiles_file, index=False)

    print('='*50)
    # Save the trained SmilesDecoder
    smiles_decoder.save_model(args.saved_smiles_decoder)
    print('Trained SmilesDecoder is saved in {}'.format(args.saved_smiles_decoder))

    return smiles_decoder



































