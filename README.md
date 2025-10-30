# CSINTSY MCO2 PinoyBot

**Repo for CSINTSY MCO2 PinoyBot**

PinoyBot is a word-level **language identifier** designed for Filipino-English code-switched text. This supervised machine learning classifier can accurately label each word in a sentence as **Filipino, English, code-switched, named entity, number, symbol, abbreviation, expression, or unknown**, helping improve NLP applications like sentiment analysis, machine translation, and speech recognition in multilingual Filipino contexts.

## Features

- Detects **Filipino** words, including those with affixes, infixes, and reduplication.
- Detects **English** words in Filipino contexts.
- Handles **intra-word code-switching** (e.g., `naglunch`, `pina-explain`).
- Recognizes **named entities**, numbers, symbols, abbreviations, onomatopoeic expressions, and unknown words.
- Built on a **validated dataset** sourced from historical Filipino-English texts (CoHFiE corpus).

## Dataset

The dataset was annotated and validated manually by students using preliminary LLM-generated tags. Each word is labeled according to the following categories:

- **Fil** – Purely Filipino words
- **Eng** – English words
- **CS** – Code-switched words
- **NE** – Named entities
- **Num** – Numbers
- **Sym** – Symbols and punctuation
- **Abb** – Abbreviations
- **Expr** – Onomatopoeic expressions
- **Unk** – Unknown or ambiguous words

## How to run

Git Clone the csintsy-mco2-pinoybot repository

```
git@github.com:mbchavez27/csintsy-mco2-pinoybot.git

```

Execute the following commands to install the dependencies

```
python -m venv venv       # create their own virtualenv
source venv/bin/activate  # activate it (Linux/macOS)
venv\Scripts\activate     # activate it (Windows)
pip install -r requirements.txt
```

Run the model first

```
python models/language_rf_model.py
```

Run the pinoybot

```
python pinoybot.py
```
