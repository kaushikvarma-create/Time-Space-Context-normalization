"""Object-name taxonomy: raw label -> canonical name, and is_a ancestors for roll-ups.

Two kinds of link:
  same : synonyms. Materialised as object_alias rows (alias -> canonical) and registry.merged_into.
  is_a : 'backpack is_a bag'. Both names stay as their own dimension; the parent also gets a roll-up
         dimension that sums its descendants.
Resolution is a pure function of the stored links, so changing a link and re-resolving re-maps all history.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


def spaced(label: str) -> str:
    """Normal form for matching: lowercase, single spaces, underscores as spaces."""
    return " ".join(label.strip().lower().replace("_", " ").split())


def snake(label: str) -> str:
    return spaced(label).replace(" ", "_")


def singular_candidates(k: str) -> list[str]:
    out = [k]
    if k.endswith("ies"):
        out.append(k[:-3] + "y")
    if k.endswith("es"):
        out.append(k[:-2])
    if k.endswith("s") and not k.endswith("ss"):
        out.append(k[:-1])
    return out


@dataclass
class Taxonomy:
    known: set[str] = field(default_factory=set)           # registry object types (canonical or merged)
    alias: dict[str, str] = field(default_factory=dict)    # spaced alias -> object type
    merged_into: dict[str, str] = field(default_factory=dict)
    parents: dict[str, set[str]] = field(default_factory=dict)   # approved is_a, canonical names

    def _follow(self, name: str) -> str:
        seen = set()
        while name in self.merged_into and name not in seen:
            seen.add(name)
            name = self.merged_into[name]
        return name

    def lookup(self, raw: str) -> str | None:
        """Canonical name for a raw label, or None if the label is unknown."""
        k = spaced(raw)
        for cand in singular_candidates(k):
            s = cand.replace(" ", "_")
            if s in self.known:
                return self._follow(s)
            if cand in self.alias:
                return self._follow(self.alias[cand])
        return None

    def resolve(self, raw: str) -> str:
        """Canonical name; unknown labels resolve to their own snake_case form."""
        return self.lookup(raw) or snake(raw)

    def ancestors(self, name: str) -> set[str]:
        """All is_a ancestors (transitive, cycle-safe), excluding the name itself."""
        out: set[str] = set()
        stack = list(self.parents.get(name, ()))
        while stack:
            p = self._follow(stack.pop())
            if p != name and p not in out:
                out.add(p)
                stack.extend(self.parents.get(p, ()))
        return out


def load_taxonomy(conn: sqlite3.Connection) -> Taxonomy:
    t = Taxonomy()
    for obj, merged in conn.execute("SELECT object_type, merged_into FROM object_registry"):
        t.known.add(obj)
        if merged:
            t.merged_into[obj] = merged
    t.alias = {a: o for a, o in conn.execute("SELECT alias, object_type FROM object_alias")}
    for child, parent in conn.execute(
            "SELECT child, parent FROM object_links WHERE relation='is_a' AND status='approved'"):
        c, p = t._follow(child), t._follow(parent)
        if c != p:
            t.parents.setdefault(c, set()).add(p)
    return t


def reresolve_observations(conn: sqlite3.Connection) -> int:
    """Recompute canonical names of all observations and relations from their raw labels (caller holds the
    transaction). Returns the number of changed rows."""
    tax = load_taxonomy(conn)
    changed = 0
    for obs_id, raw, current in conn.execute("SELECT obs_id, raw_label, object_type FROM observations "
                                             "WHERE raw_label IS NOT NULL").fetchall():
        new = tax.lookup(raw)
        if new and new != current:
            conn.execute("UPDATE observations SET object_type=? WHERE obs_id=?", (new, obs_id))
            changed += 1
    for rel_id, rs, ro, s, o in conn.execute("SELECT rel_id, raw_subject, raw_object, subject, object FROM relations "
                                             "WHERE raw_subject IS NOT NULL").fetchall():
        ns, no = tax.lookup(rs) or s, tax.lookup(ro) or o
        if (ns, no) != (s, o):
            conn.execute("UPDATE relations SET subject=?, object=? WHERE rel_id=?", (ns, no, rel_id))
            changed += 1
    return changed
