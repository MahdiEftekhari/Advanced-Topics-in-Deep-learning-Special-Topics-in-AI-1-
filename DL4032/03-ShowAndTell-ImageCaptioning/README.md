# Show and Tell: Image Captioning

## Objective
Implement an image captioning system that pairs a CNN encoder with an RNN decoder, following the architecture introduced in *Show and Tell: A Neural Image Caption Generator* (Vinyals et al., 2015). Students extract visual features from images using a pretrained CNN backbone, then generate natural-language captions word-by-word with a recurrent decoder trained via teacher forcing.

## Background / Paper
- **Paper:** Show and Tell: A Neural Image Caption Generator — Vinyals et al., CVPR 2015 (arXiv:1411.4555)
- **Dataset:** Flickr8k (~8,000 images, 5 human-written captions each)
- **Core idea:** the encoder–decoder pattern from neural machine translation, repurposed for image → text generation — a CNN stands in for the source-language encoder, feeding its features into an RNN decoder that generates the caption sequence.

## Files in this folder
This folder is a pointer, not a copy — the full assignment lives in its own repo so it stays a single source of truth:

**[github.com/Mound21k/image-captioning](https://github.com/Mound21k/image-captioning)**

That repo is organized as:
```
image_captioning_assignment/
├── data/            # Flickr8k download + preprocessing script
├── models/          # encoder.py, decoder.py, caption_model.py (TODOs live here)
├── utils/           # dataset.py, vocabulary.py, trainer.py, metrics.py (TODOs live here)
├── notebooks/       # 4-part sequence: exploration → feature extraction → training → evaluation
├── Solution/         # reference implementation
├── DL4032_HW03.pdf  # original assignment handout
└── requirements.txt
```

## How to attempt it
1. Clone the linked repo and set up the environment (`pip install -r requirements.txt`); download Flickr8k via the provided script.
2. Work through the four notebooks in order, filling in the `TODO`-marked sections in `models/` and `utils/` as you go.
3. Pick a CNN backbone for the encoder (ResNet18, ResNet50, or MobileNetV2), then implement the LSTM/GRU decoder — word embeddings, teacher forcing during training, and greedy or beam-search decoding at inference.
4. Evaluate with BLEU. A correct implementation should land around **BLEU-1 ≈ 0.60–0.65** and **BLEU-4 ≈ 0.20–0.25**.
5. Optional extensions once the core pipeline works: visual attention, a Transformer decoder, scaling up to MS COCO or Flickr30k, additional metrics (CIDEr, METEOR), or fine-tuning the CNN encoder end-to-end.

## Reference solution
A completed reference implementation lives in the `Solution/` folder of the linked repo. Worth attempting the TODOs yourself first — wiring up the encoder, decoder, and training loop by hand is where the paper's ideas actually click.
