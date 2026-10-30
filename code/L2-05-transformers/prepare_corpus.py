"""Download Romeo and Juliet from Project Gutenberg and save a clean copy.

The play is in the public domain. Project Gutenberg's license page says that
once the Project Gutenberg license and all references to Project Gutenberg
are stripped, the remaining public-domain text can be used freely. This
script keeps only the play itself and normalizes typographic quotes and
line endings so the character vocabulary stays small.

The cleaned file is already in the repository (data/romeo_and_juliet.txt);
run this only to reproduce it.
"""
import pathlib
import urllib.request

URL = "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
OUT = pathlib.Path(__file__).parent / "data" / "romeo_and_juliet.txt"

raw = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
raw = raw.replace("\r\n", "\n")
start = raw.index("THE TRAGEDY OF ROMEO AND JULIET")
end = raw.index("*** END OF THE PROJECT GUTENBERG EBOOK")
text = raw[start:end].strip() + "\n"

# Map curly quotes and dashes to ASCII; drop the stage-direction underscores.
for a, b in {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
             "\u2014": "--", "\u2013": "-", "_": ""}.items():
    text = text.replace(a, b)
assert "Gutenberg" not in text, "a Project Gutenberg reference survived"

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(text, encoding="utf-8")
print(f"saved {OUT.name}: {len(text):,} characters, "
      f"{len(set(text))} distinct, {text.count(chr(10)):,} lines")
print("non-ASCII characters left:", sorted({c for c in text if ord(c) > 127}))
