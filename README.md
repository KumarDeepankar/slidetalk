# SlideTalk

Ask a question about an H&E whole-slide image and get an answer in words.

A frozen GPT-2 learns to look at the slide through 4 gated cross-attention layers. Only those layers and
one projection are trained (10M parameters), in about 8 minutes on an M3 MacBook. It is a small teaching
prototype of the idea behind slide-language models such as Prov-GigaPath and Flamingo.

![SlideTalk architecture](docs/slidetalk.png)

## Results

42 slides from patients never seen in training (5 TCGA subtypes: lung adenocarcinoma, lung squamous,
clear cell, papillary and chromophobe renal cell carcinoma).

| Question | Correct |
|---|---|
| Which organ? | 95% |
| Which cancer subtype? (5 options, chance 20%) | 76% |
| Yes / no about the subtype | 89% |
| Pick between the subtypes of that organ | 79% |

With cross-attention switched off, GPT-2 gives every patient the same answer; shown another patient's
slide, the answer follows the slide shown. So the answers come from the slide.

## Run it

Python 3.11. Slides are downloaded from the public [NIH GDC](https://portal.gdc.cancer.gov/) (TCGA,
open access); Phikon and GPT-2 come from Hugging Face.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests -q              # model checks, no data needed
python scripts/extract_features.py     # download, tile and embed the 300 slides in data/manifest.csv
python scripts/train.py                # train cross-attention (about 8 min on an M3)
python scripts/test.py                 # accuracy on held-out patients + controls
python scripts/chat.py                 # ask questions about a test slide
```

`extract_features.py` streams one slide at a time (download, embed, delete), so disk use stays at a
couple of slides; it resumes where it stopped. `--limit 25` gives a quick start with fewer slides.
Each chat answer also saves an attention map (which tiles the answer looked at) in `runs/slidetalk/`.

## Layout

```
slidetalk/model.py         gated cross-attention + frozen GPT-2
slidetalk/data.py          labels, questions, tile sampling
slidetalk/evaluate.py      asking questions and scoring answers
slidetalk/prep/            GDC download, tissue tiling, Phikon tile encoder
scripts/                   extract_features, train, test, chat
data/manifest.csv          the 300 slides used (60 per subtype)
data/splits.csv            train / val / test, split by patient
```
