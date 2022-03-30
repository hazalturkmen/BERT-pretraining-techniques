import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--corpus",
    default=None,
    metavar="path",
    type=str,
    required=True,
    help="corpus path",
)
parser.add_argument(
    "--out",
    default=None,
    metavar="path",
    type=str,
    required=True,
    help="lower corpus path",
)
args = parser.parse_args()


def preproc(sentence):
    """

    :param string:
    :return:string
    """
    sentence = sentence.strip()
    sentence = sentence.replace('İ', 'i')
    sentence = sentence.replace('Ü', 'ü')
    sentence = sentence.replace('Ö', 'ö')
    sentence = sentence.replace('Ç', 'ç')

    sentence = sentence.lower()

    return sentence


def readInChunks(fileObj, chunkSize=2048):
    while True:
        data = fileObj.read(chunkSize)
        if not data:
            break
        yield data


print("Pre-processing medical text...")

f = open(args.corpus)
fo = open(args.out, 'w', encoding='utf-8')
for chunk in readInChunks(f):
    fo.write(preproc(chunk))

f.close()
fo.close()
