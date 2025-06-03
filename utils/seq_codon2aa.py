import sys
from Bio import SeqIO
from Bio.Seq import Seq
# add path to the python path
sys.path.append('/mnt/disk90/user/jiayili/codon_optimization_steering/utils')
from calculator.CPB.mapping import codon_to_aa

def translate_mrna_to_protein(mrna_seq):
    # Clean and uppercase input
    mrna_seq = mrna_seq.upper().replace("\n", "").replace(" ", "")
    
    # Replace DNA characters (T -> U) if needed
    mrna_seq = mrna_seq.replace("T", "U")

    protein = ""
    for i in range(0, len(mrna_seq) - 2, 3):
        codon = mrna_seq[i:i+3]
        amino_acid = codon_to_aa.get(codon, "")
        if amino_acid == "STOP":
            amino_acid = "*"  
        protein += amino_acid
    return protein

def process_fasta(input_file, output_file):
    """output file is not a fasta file, but a text file"""
    with open(output_file, "w") as out_handle:
        for record in SeqIO.parse(input_file, "fasta"):
            protein_seq = translate_mrna_to_protein(str(record.seq))
            if protein_seq:  # Only write if translation is successful
                out_handle.write(f">{record.id}_translated\n{protein_seq}\n")

if __name__ == "__main__":
    input_file = '/mnt/disk90/user/jiayili/codon_optimization_steering/data/bins/bin_930-1239/bin_930-1239.fa'
    output_file = '/mnt/disk90/user/jiayili/codon_optimization_steering/data/bins/bin_930-1239/translated_protein'
    process_fasta(input_file, output_file)
