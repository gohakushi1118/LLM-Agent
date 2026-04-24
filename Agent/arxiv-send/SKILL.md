---
name: arxiv-send
description: Retrieve a paper from arXiv by ID or keyword and email it to a recipient using himalaya. Combines the arxiv and himalaya skills into a single workflow.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Research, Arxiv, Email, Academic, Science]
    related_skills: [research/arxiv, email/himalaya]
prerequisites:
  commands: [himalaya]
---

# arXiv Paper Send

Fetch paper metadata from arXiv and send it to a recipient via email.

## Parameters

- `paper_id_or_title`: arXiv ID (e.g. `2303.12712`) or search keywords
- `recipient_email`: destination email address
- `custom_message`: (optional) personal note prepended to the email body

## Workflow

### Step 1 — Fetch paper metadata

**By arXiv ID:**

```bash
curl -s "https://export.arxiv.org/api/query?id_list=2303.12712" | python3 -c "
import sys, xml.etree.ElementTree as ET
ns = {'a': 'http://www.w3.org/2005/Atom'}
root = ET.parse(sys.stdin).getroot()
entry = root.find('a:entry', ns)
if entry is None: sys.exit('Paper not found')
title   = entry.find('a:title', ns).text.strip().replace('\n', ' ')
authors = ', '.join(a.find('a:name', ns).text for a in entry.findall('a:author', ns))
summary = entry.find('a:summary', ns).text.strip()
arxiv_id = entry.find('a:id', ns).text.strip().split('/abs/')[-1]
pdf_url = f'https://arxiv.org/pdf/{arxiv_id}'
print(f'TITLE: {title}')
print(f'AUTHORS: {authors}')
print(f'PDF: {pdf_url}')
print(f'ABSTRACT: {summary}')
"
```

**By keyword (use top result):**

```bash
curl -s "https://export.arxiv.org/api/query?search_query=all:KEYWORD&max_results=1&sortBy=relevance" | python3 -c "
import sys, xml.etree.ElementTree as ET
ns = {'a': 'http://www.w3.org/2005/Atom'}
root = ET.parse(sys.stdin).getroot()
entry = root.find('a:entry', ns)
if entry is None: sys.exit('Paper not found')
title   = entry.find('a:title', ns).text.strip().replace('\n', ' ')
authors = ', '.join(a.find('a:name', ns).text for a in entry.findall('a:author', ns))
summary = entry.find('a:summary', ns).text.strip()
arxiv_id = entry.find('a:id', ns).text.strip().split('/abs/')[-1]
pdf_url = f'https://arxiv.org/pdf/{arxiv_id}'
print(f'TITLE: {title}')
print(f'AUTHORS: {authors}')
print(f'PDF: {pdf_url}')
print(f'ABSTRACT: {summary}')
"
```

### Step 2 — Send email via himalaya

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: recipient@example.com
Subject: Research Paper: {TITLE}

{CUSTOM_MESSAGE}

---
Title:   {TITLE}
Authors: {AUTHORS}
PDF:     {PDF_URL}

Abstract:
{ABSTRACT}

---
Sent via Hermes Agent
EOF
```

## Example Phrases

- "Send paper 2303.12712 to alice@example.com"
- "Email the paper 'Attention Is All You Need' to bob@example.com with the note: 'Thought you'd find this useful.'"
- "Find the latest GRPO paper and send it to my collaborator at carol@example.com"

## Notes

- Check the abstract for "withdrawn" or "retracted" before sending — arXiv papers can be withdrawn after submission.
- The himalaya `template send` command reads from stdin; always use heredoc (`<< 'EOF'`) to avoid shell interpolation issues.
- For the sender address, use the email configured in `~/.config/himalaya/config.toml`.