# Implementation of Gx2Mol 
A PyTorch implementation of “Gx2Mol: De Novo Generation of Hit-like Molecules from Gene Expression Profiles via Deep Learning“.
The paper is under review by Neural Computing and Applications. 

![Overview of GxVAEs](https://github.com/naruto7283/Gx2Mol/blob/main/gx2mol.png)

## Objectives 
GxVAEs aim to
- generate hit-like molecules from gene expression profiles.
- generate therapeutic molecules from patients’ disease profiles.

## Environment Installation
Execute the following command:
```
$ conda env create -n gx2mol_env -f gx2mol_env.yml
$ source activate gx2mol_env 
```

## File Description

- **datasets**
    - LINCS/mcf7.csv: The training and validation datasets, which consist of gene expression profiles of the MCF7 cell line treated with 13,755 molecules, were used.
    - tools floder
- **main.py:**: Define the main function for training the GeneVAE and SmilesDecoder models.
- **GeneVAE.py**: Defines a VAE model for extracting features from gene expression profiles.
- **train_gene_vae.py**: Code for training the GeneVAE model.
- **SmilesDecoder.py**: Defines a decoder model for generating SMILES strings with extracted gene features.
- **train_smiles_vae.py**: Code for training the SmilesDecoder model.
- **utils.py**: Defines other functions used in Gx2Mol.

## Experimental Reproduction

  - **STEP 1**: Pretrain GeneVAE:
  ``` 
  $ python main.py --train_gene_vae --cell_name 'mcf7'
  ```
  - **STEP 2**: Test the trained GeneVAE:
  ```
  $ python main.py --test_gene_vae --cell_name 'mcf7'
  ```
  - **STEP 3**: Train SmilesDecoder: 
  ```  
  $ python main.py --train_smiles_decoder 
  ```
  - **STEP 4**: Test SmilesDecoder: 
  ```
  $ python main.py --test_smiles_decoder
  ```
  - **STEP 5**: Generate molecules for the 10 ligands: 
  ```
  $ python main.py --generation --protein_name 'AKT1'
  ```	
  - **STEP 6**: Calculate Tanimoto similarity between a source ligand and generated SMILES strings: 
  ```
  $ python main.py --calculate_tanimoto 
  ```
