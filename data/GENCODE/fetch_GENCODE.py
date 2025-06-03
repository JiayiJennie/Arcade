from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_fasta", type=str, help="Input fasta file")
parser.add_argument("-o", "--output_fasta", type=str, help="Output fasta file")
parser.add_argument("-m", "--max_len", type=int, default=10000, help="Maximum length of CDS sequence")
args = parser.parse_args()

input_fasta = args.input_fasta
output_fasta = args.output_fasta
max_len = args.max_len

num_cds = 0
with open(output_fasta, "w") as output_handle:
    for record in tqdm(SeqIO.parse(input_fasta, "fasta")):
        record_split = record.description.split("|")
        new_id_lst = record_split[:6]
        cds_info = [x for x in record_split if x.startswith("CDS")][-1]
        pattern = r'(\d+)-(\d+)'
        cds_len_lst = re.search(pattern, cds_info)
        start,end = cds_len_lst.groups()
        start,end = int(start), int(end)
        cds_seq = record.seq[start-1:end]
        assert len(cds_seq) == end-start+1
        if len(cds_seq) > max_len or len(cds_seq) % 3 != 0:
            continue
        new_id = "|".join(new_id_lst)+"|CDS_len:"+str(len(cds_seq))+"|"

        ### write new record to output file
        new_record = SeqRecord(cds_seq, id=new_id, description="")
        SeqIO.write(new_record, output_handle, "fasta")
        num_cds += 1

print(f"Number of sequences with CDS length < {max_len}: {num_cds}")







