import RNA
import os
import fnmatch
import pandas as pd
import numpy as np
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from Bio import SeqIO
from tqdm import tqdm

# input: cDNA sequences
def mfe_calculation(seq):
    """
    calculate MFE for given seq

    parameters:
    - seq: cDNA sequence

    return:
    - MFE value
    """
    # fc = RNA.fold_compound(seq)
    # (ss, mfe) = fc.mfe()
    # return mfe
    mfe_structure, fc = RNA.fold(seq)
    return fc, mfe_structure

def gc_content(seq):
    """
    calculate GC content for given seq

    parameters:
    - seq: cDNA sequence

    return:
    - GC content value
    """
    gc_count = seq.count('G') + seq.count('C')
    total_count = len(seq)
    if total_count == 0:
        return 0
    return gc_count / total_count

def CpG_density(seq):
    """
    calculate CpG per 100 bp for given seq (https://pubmed.ncbi.nlm.nih.gov/33975521/)

    parameters:
    - seq: codon sequence

    return:
    - CpG density value
    """
    cpg_count = seq.count('CG')
    total_count = len(seq) 
    if total_count == 0:
        return 0
    return cpg_count / total_count * 100 if total_count > 0 else 0

def UpA_density(seq):
    """
    calculate UpA density for given seq

    parameters:
    - seq: codon sequence

    return:
    - UpA density value
    """
    seq = seq.replace('T', 'U')
    upa_count = seq.count('UA')
    total_count = len(seq)  
    if total_count == 0:
        return 0
    return upa_count / total_count * 100 if total_count > 0 else 0

# transfer table to dict
def table_to_dict(table, output=None, flag=False):
    """
    transfer given table to dict for calculation

    parameters:
    - table: codon usage frequency table
    - output: output path
    - flag: whether to save the result

    return:
    None
    """
    data = pd.read_csv(table, sep='\t')
    result = dict()
    max_result = dict()
    for i in range(len(data['Amino acid'])):
        if data['Amino acid'][i] not in max_result:
            max_result[data['Amino acid'][i]] = data['Fraction'][i]
        else:
            if data['Fraction'][i] > max_result[data['Amino acid'][i]]:
                max_result[data['Amino acid'][i]] = data['Fraction'][i]
        result[data['CODON'][i]] = [data['Amino acid'][i],data['Fraction'][i]]
    name = 'codon_adaptiveness.json'
    # if flag:
    #     with open(output+'/'+name, 'wb') as file:
    #         pickle.dump(result, file)
    #     with open(output+'/'+'max_'+name, 'wb') as file:
    #         pickle.dump(max_result, file)
    if flag:
        with open(output+'/'+name, 'w') as file:
            json.dump(result, file)
        with open(output+'/'+'max_'+name, 'w') as file:
            json.dump(max_result, file)

def cai_calculation(seq, table, max_table=None):
    """
    calculate Codon Adaptation Index (CAI) for given seq

    parameters:
    - seq: cDNA sequence
    - table: codon usage frequency table

    return:
    - CAI value
    """
    table = json.load(open(table, 'rb'))
    
    if max_table == None:
        pass
    else:
        max_table = json.load(open(max_table, 'rb'))

        seq = ''.join(seq.split())
        codons = [seq[i:i+3] for i in range(0, len(seq), 3) if len(seq[i:i+3]) == 3]
        codons = [codon.replace('U', 'T') for codon in codons]

        adaptiveness_values = []
        for codon in codons:
            freq = table[codon][1]
            max_freq = max_table[table[codon][0]]
            adaptiveness_values.append(freq/max_freq)
        
        if any(value == 0 for value in adaptiveness_values):
            print('There is a zero in the codon table!!!')
            return 0

    CAI = np.prod(adaptiveness_values) ** (1.0 / len(adaptiveness_values))

   
    return CAI

def cpb_calculation(seq, cps_table):
    """
    calculate Codon Pair Bias (CPB) for given seq

    parameters:
    - seq: cDNA sequence
    - cps_table: codon pair usage frequency table

    return:
    - CPB value
    """
    cps_table = pd.read_csv(cps_table)
    cps_table = dict(zip(cps_table['CODON_PAIR'], cps_table['CPS']))
    seq = ''.join(seq.split())
    codons = [seq[i:i+3] for i in range(0, len(seq), 3) if len(seq[i:i+3]) == 3]
    codons = [codon.replace('U', 'T') for codon in codons]

    cpb_values = []
    for i in range(len(codons)-1):
        cp = codons[i] + codons[i+1]
        if cp in cps_table:
            cpb_values.append(cps_table[cp])
        else:
            raise ValueError(f"Codon pair {cp} not found in codon pair usage table.")
    
    CPB = np.mean(cpb_values)
    return CPB


# input: directory
def sequences_from_dir(path, pattern=".fasta", output_file="mfe_table.csv"):
    seqs = []
    names = []

    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, f"*{pattern}"): 
            file_path = os.path.join(path, filename)
            for record in SeqIO.parse(file_path, "fasta"):
                seqs.append(str(record.seq))
                names.append(record.id)
    return seqs, names

    # seqs = []
    # names = []

    # for filename in os.listdir(path):
    #     if fnmatch.fnmatch(filename, f"*{pattern}"): 
    #         file_path = os.path.join(path, filename)
    #         with open(file_path, 'r') as f:
    #             seq = ''.join(line.strip() for line in f if not line.startswith(">"))
    #             seqs.append(seq)
    #             names.append(filename.replace(pattern, ""))
    # return seqs, names

# input file
def sequences_from_file(file):
    seqs = []
    names = []
    for record in SeqIO.parse(file, "fasta"):
        seqs.append(str(record.seq))
        names.append(record.id)
    return seqs, names

    # seqs = []
    # names = []
    # current_seq = []
    # with open(file, 'r') as f:
    #     for line in f:
    #         line = line.strip()
    #         if line.startswith(">"):
    #             if current_seq:  
    #                 seqs.append(''.join(current_seq))
    #                 current_seq = []

    #             names.append(line.split("|")[1])
    #         else:
    #             current_seq.append(line)

    #     if current_seq:
    #         seqs.append(''.join(current_seq))
    # return seqs, names

# Parallel processing function
def process_sequence_mfe(seq, name, structure=False):
    if structure:
        mfe, mfe_structure = mfe_calculation(seq)
        return name, mfe, mfe_structure, len(seq)
    else: 
        mfe, mfe_structure = mfe_calculation(seq)
        return name, mfe, len(seq)

def process_sequence_both(seq, name, table, mtable, cps_table):
    mfe, mfe_structure = mfe_calculation(seq)
    cai = cai_calculation(seq, table, mtable)
    GC_content = gc_content(seq)
    if cps_table is not None:
        cpb = cpb_calculation(seq, cps_table)
        return name, mfe_structure, mfe, cai, len(seq), GC_content, cpb
    else:
        return name, mfe_structure, mfe, cai, len(seq), GC_content


if __name__ == '__main__':
    # initialize
    parser = argparse.ArgumentParser(description="A simple example to demonstrate argparse.")

    # parameters
    parser.add_argument('-i', type=str, required=False, help="Path to the input file/dir")
    parser.add_argument('-o', type=str, required=False, help="Path to the output file")
    parser.add_argument('-f', type=str, required=False, help="Function to use, MFE or CAI or both")
    parser.add_argument('-table', type=str, required=False, help="Codon table file")
    parser.add_argument('-mtable', type=str, required=False, help="Max codon table file")
    parser.add_argument('-cps_table', type=str, required=False, help="Codon pair usage table file")

    parser.add_argument('-itable', type=str, required=False, help="Path to the codon table")
    parser.add_argument('-otable', type=str, required=False, help="Path to save codon table dictionary")
    parser.add_argument('-p', action='store_true', default=False, required=False, help="preprocess the given codon table")
    args = parser.parse_args()

    if args.p:
        if not args.itable or not args.otable:
            print("Please provide the path to the codon table and the output file")
            sys.exit(1)
        print("Preprocessing the codon table")
        print("Saving the codon table to", args.otable)
        print("Saving max codon table to", args.otable)
        table_to_dict(args.itable, args.otable, True)
        print('\n=============== No Bug No Error, Finished!!! ===============')
        sys.exit(1)
        
    if not args.i or not args.o or not args.f:
        print("Please provide the input file/dir, output file and the function to use")
        sys.exit(1)

    path = args.i
    output = args.o
    os.makedirs(os.path.dirname(args.o), exist_ok=True)

    
    if os.path.isfile(path):
        # print("Processing single file from", path)
        print("Processing fasta file from", path)
        seqs, names = sequences_from_file(path)
    else:
        print("Processing directory from", path)
        seqs, names = sequences_from_dir(path)
    
    if args.f == 'both': # also with MFE structure and GC content
        mfe_data = []
        cai_data = []
        # Use ProcessPoolExecutor for parallel computing
        num_workers = os.cpu_count() // 2
        print(f"Using {num_workers} CPU cores for parallel computing")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            process_sequence_both_with_args = partial(process_sequence_both, table=args.table, mtable=args.mtable, cps_table=args.cps_table)
            result = list(tqdm(executor.map(process_sequence_both_with_args, seqs, names), 
                               total=len(seqs),
                               desc="Calculating MFE and CAI"))

        # df = pd.DataFrame(result, columns=["ID", "MFE", "CAI","Length"])
        if args.cps_table is not None:
            df = pd.DataFrame(result, columns=["ID", "MFE_Structure", "MFE", "CAI","Length", "GC_Content", "CPB"])
        else:
            df = pd.DataFrame(result, columns=["ID", "MFE_Structure", "MFE", "CAI","Length", "GC_Content"])
        
    elif args.f == 'MFE':
        mfe_data = []
        # Use ProcessPoolExecutor for parallel computing
        num_workers = os.cpu_count() // 2  # Use half of available CPU cores
        print(f"Using {num_workers} CPU cores for parallel computing")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            result = list(tqdm(executor.map(process_sequence_mfe, seqs, names), 
                               total=len(seqs),
                               desc="Calculating MFE"))
        df = pd.DataFrame(result, columns=["ID", "MFE", "Length"])

    elif args.f == 'CAI':
        result = []
        for seq, name in tqdm(zip(seqs, names), total=len(seqs)):
            cai = cai_calculation(seq, args.table, args.mtable)
            result.append([name, cai, len(seq)])
        df = pd.DataFrame(result, columns=["ID", "CAI","Length"])
    
    df.to_csv(output, index=False)
    # print(df)
    print(f"result saved to {output}")   
    print('\n=============== No Bug No Error, Finished!!! ===========')