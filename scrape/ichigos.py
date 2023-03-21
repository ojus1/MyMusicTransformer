import requests
from bs4 import BeautifulSoup
import re
import os
from joblib import Parallel, delayed
from tqdm import tqdm
os.makedirs("data/ichigos", exist_ok=True)

suffix = [
    'others',
    'fi',
    *[item for item in 'abcdefghijklmnopqrstuvwxyz']
]

page_url = 'https://ichigos.com/sheets/'
required_names = [
    ' for piano',
    '"for piano"',
    "'for piano'",
    "'piano solo'",
    "('for piano')",
    '(for piano)',
    'a piano arrangement, for the advanced pianist',
    'arranged for piano solo',
    'for easier piano',
    'for easy piano',
    'for easy piano or mallets',
    'for jazz piano',
    'for piano',
    'for piano ',
    'for piano & flute',
    'for piano (duet not included)',
    'for piano (full rock version)',
    'for piano (intermediate)',
    'for piano (trust me! it sounds better on piano -dedicated to prissyrox4vr)',
    'for piano (tv size)',
    'for piano - [simplified]',
    'for piano - midi made from a pdf from the net...',
    'for piano 1',
    'for piano [exercise 7th chords]',
    'for piano [exercise chords]',
    'for piano in c# major and easy c major',
    'for piano only',
    'for piano or harp',
    'for piano or synthesizer ',
    'for piano solo',
    'for piano solo or lead sheet',
    'for piano(based on bgm in ep8)',
    'for piano, credits to kerengi on fiverr',
    "for piano, from piano stories best '88-'08",
    'for piano, transcribed for sonickku',
    'for piano, vibraphone',
    'for piano, w/ optional ostinato',
    'for piano; original key',
    'for recoder/ piano easy',
    'for simple piano',
    'for solo piano',
    'from final fantasy xv original soundtrack: piano arrangements',
    'jazz piano',
    'piano',
    'piano ',
    'piano (easier c major)',
    'piano (original ab major)',
    'piano (resubmit)',
    'piano - advanced',
    'piano arrangement',
    'piano collection / medley',
    'piano collections: moonlit melodies',
    'piano cover',
    'piano cover by eriol',
    'piano medley',
    'piano melody',
    'piano sequence of song',
    'piano sheet',
    'piano solo',
    'piano solo (revised)',
    'piano solo with variations',
    'piano version',
    'piano, arranged',
    'sheet music for piano',
    'simple piano',
    'simplifed piano',
    'solo piano',
    'strictly for piano',
]

midi_request_prefix = "https://ichigos.com"
def parse_pages():
    print("Parsing URLs")
    links = []
    for s in tqdm(suffix):
        soup = BeautifulSoup(requests.get(f"{page_url}/{s}").content, "html.parser")
        possible_tags = soup.find_all(href=re.compile(r"type\=midi"))
        
        for item in possible_tags:
            n = item.find_previous_sibling("i").text.lower()
            if n in required_names:
                links.append(f"{midi_request_prefix}/{item.attrs['href']}")
    return links

def download(link, i):
    path = os.path.join("data/ichigos", f"{i}_ichigo.mid")

    with open(path, "wb") as f:
        f.write(requests.get(link).content)

links = parse_pages()
print("Downloading MIDIs")
Parallel(n_jobs=-1, prefer="processes")(delayed(download)(l, i) for i, l in tqdm(enumerate(links), total=len(links)))

