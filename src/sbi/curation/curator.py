"""Keep the object vocabulary clean as it grows: find synonyms ('same') and hierarchy ('is_a') links.

Method (validated 2026-11-13 on 79 candidate pairs from real VLM output: 11 links, all correct; the one-shot
"group these names" and single-question variants produced nonsense such as 'desk is_a keyboard'):
  1. shortlist name pairs by text-embedding cosine >= min_similarity
  2. ask two independent yes/no questions: "is every A a B?" and "is every B an A?"
       yes/yes -> same     yes/no -> A is_a B     no/yes -> B is_a A     no/no -> no link
  3. store the link with evidence; if auto-approved, apply it and re-map history from raw labels.
Everything is reversible (set_link_status), and answers are cached per model so nothing is asked twice.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

from ..store.db import transaction
from ..taxonomy import reresolve_observations, spaced
from ..timeutil import now_utc
from .services import PairJudge, TextEmbedder

log = logging.getLogger(__name__)


@dataclass
class CurationReport:
    names: int = 0
    candidates: int = 0
    questions_asked: int = 0
    questions_cached: int = 0
    links: list[dict] = field(default_factory=list)
    reresolved_rows: int = 0


def _now() -> str:
    return now_utc().isoformat(timespec="seconds")


def _judge(conn: sqlite3.Connection, judge: PairJudge, a: str, b: str, report: CurationReport) -> bool:
    row = conn.execute("SELECT answer FROM label_judgments WHERE a=? AND b=? AND model=?", (a, b, judge.name)).fetchone()
    if row is not None:
        report.questions_cached += 1
        return bool(row[0])
    ans = judge.is_every(spaced(a), spaced(b))                 # slow model call: outside any transaction
    report.questions_asked += 1
    with transaction(conn):
        conn.execute("INSERT OR REPLACE INTO label_judgments(a,b,model,answer,created_utc) VALUES(?,?,?,?,?)",
                     (a, b, judge.name, int(ans), _now()))
    return ans


def _support(conn: sqlite3.Connection, name: str) -> tuple[int, int]:
    status = conn.execute("SELECT review_status FROM object_registry WHERE object_type=?", (name,)).fetchone()
    trusted = 1 if status and status[0] in ("seed", "approved") else 0
    n = conn.execute("SELECT COUNT(*) FROM observations WHERE object_type=?", (name,)).fetchone()[0]
    return trusted, n


def pick_canonical(conn: sqlite3.Connection, a: str, b: str) -> tuple[str, str]:
    """(canonical, merged): prefer configured/reviewed names, then the more observed one, then the shorter."""
    ka = (*_support(conn, a), -len(a), a)
    kb = (*_support(conn, b), -len(b), b)
    return (a, b) if ka >= kb else (b, a)


def apply_same(conn: sqlite3.Connection, child: str, parent: str) -> None:
    """Merge `child` into `parent`. Caller holds the transaction."""
    conn.execute("UPDATE object_registry SET merged_into=?, review_status='merged' WHERE object_type=?", (parent, child))
    conn.execute("UPDATE object_registry SET merged_into=? WHERE merged_into=?", (parent, child))
    conn.execute("UPDATE object_alias SET object_type=? WHERE object_type=?", (parent, child))
    conn.execute("INSERT OR REPLACE INTO object_alias(alias,object_type) VALUES(?,?)", (spaced(child), parent))


def undo_same(conn: sqlite3.Connection, child: str, parent: str) -> None:
    conn.execute("UPDATE object_registry SET merged_into=NULL, review_status='approved' WHERE object_type=? "
                 "AND merged_into=?", (child, parent))
    conn.execute("DELETE FROM object_alias WHERE alias=? AND object_type=?", (spaced(child), parent))


def _record(conn: sqlite3.Connection, child: str, parent: str, relation: str, *, approved: bool, similarity: float,
            evidence: dict, model: str) -> int:
    cur = conn.execute(
        """INSERT OR IGNORE INTO object_links(child,parent,relation,status,source,similarity,evidence_json,model,
             created_utc,decided_utc) VALUES(?,?,?,?,'curator',?,?,?,?,?)""",
        (child, parent, relation, "approved" if approved else "proposed", round(float(similarity), 4),
         json.dumps(evidence), model, _now(), _now() if approved else None))
    return int(cur.lastrowid) if cur.rowcount == 1 else 0


def curate(conn: sqlite3.Connection, embedder: TextEmbedder, judge: PairJudge, *, min_similarity: float,
           auto_approve: bool) -> CurationReport:
    rep = CurationReport()
    names = [r[0] for r in conn.execute("SELECT object_type FROM object_registry WHERE merged_into IS NULL "
                                        "ORDER BY object_type")]
    rep.names = len(names)
    if len(names) < 2:
        return rep
    vecs = embedder.embed([spaced(n) for n in names])
    sims = vecs @ vecs.T
    linked = {frozenset((c, p)) for c, p in conn.execute("SELECT child, parent FROM object_links")}
    pairs = sorted(((float(sims[i, j]), names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))
                    if sims[i, j] >= min_similarity and frozenset((names[i], names[j])) not in linked), reverse=True)
    rep.candidates = len(pairs)
    changed = False
    for sim, a, b in pairs:
        merged = {r[0] for r in conn.execute("SELECT object_type FROM object_registry WHERE merged_into IS NOT NULL")}
        if a in merged or b in merged:          # an earlier decision in this run already merged one of them
            continue
        ab, ba = _judge(conn, judge, a, b, rep), _judge(conn, judge, b, a, rep)
        if not (ab or ba):
            continue
        evidence = {"every_a_is_b": ab, "every_b_is_a": ba, "a": a, "b": b}
        with transaction(conn):
            if ab and ba:
                parent, child = pick_canonical(conn, a, b)
                lid = _record(conn, child, parent, "same", approved=auto_approve, similarity=sim, evidence=evidence,
                              model=judge.name)
                if lid and auto_approve:
                    apply_same(conn, child, parent)
                    changed = True
                rel = "same"
            else:
                child, parent = (a, b) if ab else (b, a)
                lid = _record(conn, child, parent, "is_a", approved=auto_approve, similarity=sim, evidence=evidence,
                              model=judge.name)
                changed = changed or bool(lid and auto_approve)
                rel = "is_a"
        if lid:
            rep.links.append({"link_id": lid, "child": child, "parent": parent, "relation": rel,
                              "similarity": round(sim, 3), "status": "approved" if auto_approve else "proposed"})
            log.info("link %s %s %s (cos %.2f, %s)", child, rel, parent, sim, "applied" if auto_approve else "proposed")
    if changed:
        with transaction(conn):
            rep.reresolved_rows = reresolve_observations(conn)
    return rep


def set_link_status(conn: sqlite3.Connection, link_id: int, status: str) -> dict:
    """Approve or reject a link (human decision). Applies or undoes merges and re-maps history."""
    if status not in ("approved", "rejected"):
        raise ValueError("status must be 'approved' or 'rejected'")
    with transaction(conn):
        row = conn.execute("SELECT child,parent,relation,status FROM object_links WHERE link_id=?", (link_id,)).fetchone()
        if row is None:
            raise ValueError(f"no link {link_id}")
        child, parent, relation, old = row
        if relation == "same":
            if status == "approved" and old != "approved":
                apply_same(conn, child, parent)
            elif status == "rejected" and old == "approved":
                undo_same(conn, child, parent)
        conn.execute("UPDATE object_links SET status=?, source='human', decided_utc=? WHERE link_id=?",
                     (status, _now(), link_id))
        n = reresolve_observations(conn)
    return {"link_id": link_id, "child": child, "parent": parent, "relation": relation, "status": status,
            "reresolved_rows": n}
