"""Cross-check every arXiv-linked 'Further reading' entry against arXiv's own title."""
import re, sys, glob, urllib.request, xml.etree.ElementTree as ET

entries = []  # (file, cited_title, arxiv_id)
for path in sorted(glob.glob("docs/level-*/*.md")):
    text = open(path, encoding="utf-8").read()
    tail = text.split("## Further reading")[-1]
    for line in tail.splitlines():
        if not line.startswith("- **"):
            continue
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", line)
        if not m:
            continue
        t = re.match(r'- \*\*(.*?)\*\*', line)
        cited = t.group(1) if t else ""
        quoted = re.findall(r'"([^"]+)"', cited)
        entries.append((path, quoted[-1] if quoted else cited, m.group(1)))

ids = sorted({e[2] for e in entries})
titles = {}
for i in range(0, len(ids), 40):
    chunk = ids[i:i+40]
    url = "http://export.arxiv.org/api/query?max_results=%d&id_list=%s" % (len(chunk), ",".join(chunk))
    data = urllib.request.urlopen(url, timeout=60).read()
    for entry in ET.fromstring(data).findall("{http://www.w3.org/2005/Atom}entry"):
        eid = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/abs/")[-1]
        eid = re.sub(r"v\d+$", "", eid)
        titles[eid] = " ".join(entry.find("{http://www.w3.org/2005/Atom}title").text.split())

norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " "))
bad = 0
for path, cited, aid in entries:
    real = titles.get(aid)
    if real is None:
        print("NOT FOUND ", aid, path); bad += 1; continue
    a, b = norm(cited), norm(real)
    if not (a[:40] in b or b[:40] in a):
        print("MISMATCH  ", aid, "\n   cited:", cited, "\n   arXiv:", real, "\n   in:", path); bad += 1
print(f"checked {len(entries)} entries ({len(ids)} distinct arXiv IDs); problems: {bad}")
