# GENCODE Human Release 47 (GRCh38.p14)  

## Download and Process Protein-Coding Transcript Sequences  

### 1. Visit the GENCODE Human Release 47 Website  
Visit the official **GENCODE Human Release 47 (GRCh38.p14)** website for more details:  
[https://www.gencodegenes.org/human/](https://www.gencodegenes.org/human/)  

### 2. Download the FASTA File  
Download the FASTA file containing **protein-coding transcript sequences** using the following command:  
```bash
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/gencode.v47.pc_transcripts.fa.gz
```
(This version was downloaded on March 7, 2025.)

### 3. Unzip the Downloaded File  
Use the following command to extract the `.gz` file:  

```bash
gunzip gencode.v47.pc_transcripts.fa.gz
```

### 4. Obtain the Extracted FASTA File

After unzipping, you will get the following FASTA file:
```bash
gencode.v47.pc_transcripts.fa
```

### 5. Extract CDS Sequences

The script fetch_GENCODE.py is used to extract only the CDS (coding sequence) from the FASTA file.

Usage:

Run the script as follows:
```bash
python fetch_GENCODE.py -i gencode.v47.pc_transcripts.fa -o gencode.v47.pc_transcripts_cds.fa
```

### Dataset Preprocessing Notes

1. We filter out CDS sequences longer than 3,072 nucleotides due to the maximum input length constraint of the base model CodonBERT.

2. The dataset is split into training and testing sets at a 9:1 ratio. To avoid information leakage, all transcript variants belonging to the same gene are assigned to the same split.

3. The test dataset is located at: `data/GENCODE/gencode.v47.pc_transcripts_cds_test.fa`