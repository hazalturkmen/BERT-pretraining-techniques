import argparse
import glob
import shutil
import os
from tokenizers import BertWordPieceTokenizer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--Largecorpus",
    default=None,
    metavar="path",
    type=str,
    required=True,
    help="Large corpus path",
)
parser.add_argument(
    "--Smallcorpus",
    default=None,
    metavar="path",
    type=str,
    required=True,
    help="Small corpus path",
)
parser.add_argument(
    "--out",
    default="./",
    type=str,
    help="Path to the output directory, where the files will be saved",
)
parser.add_argument(
    "--name", default="bert-wordpiece", type=str, help="The name of the output vocab files"
)

parser.add_argument(
    "--type", default="cased", type=str, help="The type of the tokenizer model(cased or uncased)"
)

args = parser.parse_args()



Sfile_size = os.path.getsize(args.Smallcorpus)
Lfile_size = os.path.getsize(args.Largecorpus)

dup = int(Lfile_size/Sfile_size)
current_directory = os.getcwd()
final_directory = current_directory+'/copy-path'
os.makedirs(final_directory)

for i in range(dup):
    pathname = os.path.join(final_directory, str(i+1)+".txt")
    shutil.copy(args.Smallcorpus, pathname)

shutil.copy(args.Largecorpus, final_directory+"/large.txt")


read_files = glob.glob(final_directory+"/"+"*.txt")

with open(final_directory+'/result.txt', 'w') as outfile:
    for fname in read_files:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
            outfile.write("\n")

if args.type != "cased":
    print("uncased")
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True,
    )
else:
    print("cased")
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )



#train tokenizer

trainer = tokenizer.train(
    final_directory+'/result.txt',
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

# Save the files
tokenizer.save_model(args.out, args.name)