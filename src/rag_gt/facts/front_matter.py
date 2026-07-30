"""Universal front-matter / boilerplate detector for extracted facts.

Content-based, NOT per-document page ranges — so any new PDF's copyright,
standards-organisation, and bibliographic boilerplate is excluded without adding
config. High precision by design: every cue is distinctive of publishing/standards
front matter, so domain technical sentences do not match.

Catches the residual leak class seen in v17: ISO/Ecma copyright pages, committee
boilerplate ("member bodies", "technical committee"), patent-rights notices,
trade-name disclaimers, and "official version" statements.
"""

from __future__ import annotations

import re

# Each pattern is a distinctive boilerplate phrase. A domain technical claim
# (welding, RL, JSON, requirements) will not contain these.
_BOILERPLATE_PATTERNS = [
    r"all rights reserved",
    r"no part of this (?:publication|document) may be reproduced",
    r"\bcopyright\b",
    r"©",
    r"\bisbn\b",
    r"\bdoi:\s",
    r"member bod(?:y|ies)\b",
    r"national standards? bod(?:y|ies)",
    r"technical committee\b",
    r"\bsubcommittee\b",
    r"the secretariat\b",
    r"patent rights?\b",
    r"identifying (?:any or all such )?patent",
    r"use of (?:the )?trade names?",
    r"does not imply (?:any )?(?:endorsement|recommendation|approval)",
    r"official (?:english )?version of (?:an?|this|the)\b",
    r"\bthe official version\b",
    r"prepared by (?:technical )?committee",
    r"in accordance with the .{0,40}directives",
    r"shall be (?:sent|addressed) to .{0,40}(?:secretariat|committee)",
    r"\bforeword\b",
    r"iso copyright office|www\.iso\.org|ecma-international\.org|www\.ecma",
    r"this (?:document|standard) supersedes",
    r"rights of use\b",
    r"adopted by (?:cen|iso|iec)\b",
    r"representation on (?:the )?technical committee",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)


def is_boilerplate(text: str) -> bool:
    """Return True when the text is publishing/standards-org front-matter boilerplate."""
    if not text:
        return False
    return bool(_BOILERPLATE_RE.search(text))
