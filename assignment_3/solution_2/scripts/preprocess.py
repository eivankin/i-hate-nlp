import json
from pathlib import Path
import warnings
import os

from spacy.attrs import ORTH, NORM

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import spacy
from spacy.tokens import DocBin, SpanGroup
import typer
from tqdm import tqdm


def main(
    input_file: Path,
    train_output: Path,
    val_output: Path,
    language: str = "ru",
    val_split: float = 0.2,
    span_key: str = "sc",
) -> None:
    """
    Preprocess the RuNNE data and save it to .spacy format.
    1. Tokenize the data with tuned tokenizer
    2. Change end boundaries (+1) so it is consistent with spacy format
    3. Save information as span groups
    4. Split to train and validation sets according to val_split value.
    5. Save data on disk.

    :param input_file: path to train.jsonl
    :param train_output: path to save train set
    :param val_output: path to save validation set
    :param language: spacy tokenizer language, "ru" by default
    :param val_split: fraction of a validation set, 20% by default
    :param span_key: where to save span info (doc.spans[span_key]), "sc" by default
    """
    nlp = spacy.blank(language)

    infixes = nlp.Defaults.infixes + [
        r"(?<!\s):(?!\s)",  # Split on ":" not preceded/followed by whitespace
        r"(?<!\s)\|(?!\s)",  # Split on "|" not preceded/followed by whitespace
        r"(?<!\s)\-(?!\s)",  # Split on "-" (the short boi) not preceded/followed by whitespace
        r"(?<!\s)\–(?!\s)",  # Split on "–" (the long boi) not preceded/followed by whitespace
        r"(?<!\s)\—(?!\s)",  # Split on "—" (the longest boi) not preceded/followed by whitespace
        r"(?<!\s)\.(?!\s)",  # Split on "." not preceded/followed by whitespace
        r"(?<!\s)\,(?!\s)",  # Split on "," not preceded/followed by whitespace
        r"(?<!\s)\/(?!\s)",  # Split on "/" not preceded/followed by whitespace
        r"(?<!\s)’(?!\s)",  # Split on "’" not preceded/followed by whitespace
        r"(?<!\s)[«»\(\)\[\]](?!\s)",  # Split on brackets and quotes not preceded/followed by whitespace
    ]
    nlp.tokenizer.infix_finditer = spacy.util.compile_infix_regex(infixes).finditer

    suffixes = nlp.Defaults.suffixes + [
        "․",  # It is a dot, but not a normal dot
        "/",
    ]
    nlp.tokenizer.suffix_search = spacy.util.compile_suffix_regex(suffixes).search

    prefixes = nlp.Defaults.prefixes + [
        "/",
    ]
    nlp.tokenizer.prefix_search = spacy.util.compile_prefix_regex(prefixes).search

    # Make some punctuation be separate tokens by overriding default rules
    nlp.tokenizer.add_special_case("''", [{ORTH: "'"}, {ORTH: "'"}])
    nlp.tokenizer.add_special_case("руб.", [{ORTH: "руб", NORM: "рубль"}, {ORTH: "."}])
    nlp.tokenizer.add_special_case(
        "долл.", [{ORTH: "долл", NORM: "доллар"}, {ORTH: "."}]
    )
    nlp.tokenizer.add_special_case("г.", [{ORTH: "г"}, {ORTH: "."}])
    nlp.tokenizer.add_special_case("гг.", [{ORTH: "гг", NORM: "годы"}, {ORTH: "."}])

    # Disable special rules regarding smileys
    nlp.tokenizer.add_special_case(":0", [{ORTH: ":"}, {ORTH: "0"}])
    nlp.tokenizer.add_special_case(":1", [{ORTH: ":"}, {ORTH: "1"}])
    nlp.tokenizer.add_special_case(":3", [{ORTH: ":"}, {ORTH: "3"}])

    all_docs = []
    with input_file.open("r", encoding="utf-8") as json_lines:
        for line in tqdm(json_lines):
            data = json.loads(line)
            text, annotations = data["sentences"], data["ners"]
            doc = nlp(text)
            spans = []
            for start, end, label in annotations:
                span = doc.char_span(start, end + 1, label=label)
                assert span is not None, (
                    text[start - 5 : start],
                    text[start : end + 1],
                    text[end + 1 : end + 6],
                )
                spans.append(span)
            doc.spans[span_key] = SpanGroup(doc, name=span_key, spans=spans)
            all_docs.append(doc)

    val_len = int(len(all_docs) * val_split)

    train_docs = all_docs[:-val_len]
    val_docs = all_docs[-val_len:]

    train_db = DocBin(docs=train_docs, store_user_data=True)
    train_db.to_disk(train_output)

    val_db = DocBin(docs=val_docs, store_user_data=True)
    val_db.to_disk(val_output)


if __name__ == "__main__":
    typer.run(main)
