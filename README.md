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
