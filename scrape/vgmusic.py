import requests
from bs4 import BeautifulSoup
import re
import os
from joblib import Parallel, delayed
from tqdm import tqdm
os.makedirs("data/vgmusic", exist_ok=True)

page_url = "http://vgmusic.com/music/other/miscellaneous/piano/"
soup = BeautifulSoup(requests.get(page_url).content, "html.parser")
# print(soup)

links = soup.find_all(href=re.compile(r".*\.mid"))
links = [item.attrs["href"] for item in links]

dl_root = "data/vgmusic/"
def download(url, midi_name):
    path = os.path.join(dl_root, f"vgmusic_{midi_name}")
    dl_url = f"{url}/{midi_name}"

    with open(path, "wb") as f:
        f.write(requests.get(dl_url).content)

Parallel(n_jobs=-1, prefer="processes")(delayed(download)(page_url, m) for m in tqdm(links))
