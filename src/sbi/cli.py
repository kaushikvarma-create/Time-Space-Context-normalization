"""Command line: python -m sbi <command>   (run from the project root with PYTHONPATH=src)

  tick        capture + process every space in config/spaces/ (what cron runs)
  migrate     apply pending database migrations
  baseline    rebuild baselines (all buckets) for one or all spaces
  reprocess   re-run Engine 1 on stored frames (e.g. after a prompt or model change)
  eval        accuracy report against human labels
  curate      find synonym ('same') and hierarchy ('is_a') links between object names, apply them, re-map history
  links       list taxonomy links            link-set ID approved|rejected   human decision (reversible)
  table       build the Chronos-ready table + column catalog for a space (CSV in data/exports/)

Exit codes: 0 ok, 1 some spaces failed, 2 fatal setup/config error.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import baseline
from .errors import SbiError
from .eval.vlm_eval import report
from .logging_setup import configure_logging
from .perception.factory import build_detector, build_scene_reader
from .pipeline import process_frame
from .settings import load_settings
from .spaces import discover_space_files, load_space
from .store.db import connect, migrate
from .tick import run_tick

log = logging.getLogger("sbi")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sbi", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".", help="project root (default: current directory)")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tick")
    t.add_argument("spaces", nargs="*", help="space YAML files (default: config/spaces/*.yaml)")
    t.add_argument("--no-vlm", action="store_true")
    sub.add_parser("migrate")
    b = sub.add_parser("baseline"); b.add_argument("--space")
    r = sub.add_parser("reprocess"); r.add_argument("--space", required=True)
    r.add_argument("--frame-id", type=int, action="append", help="repeatable; default: all processed frames")
    r.add_argument("--no-vlm", action="store_true")
    sub.add_parser("eval")
    cu = sub.add_parser("curate"); cu.add_argument("--no-auto-approve", action="store_true")
    li = sub.add_parser("links"); li.add_argument("--status", choices=["proposed", "approved", "rejected"])
    ls = sub.add_parser("link-set"); ls.add_argument("link_id", type=int)
    ls.add_argument("status", choices=["approved", "rejected"])
    ta = sub.add_parser("table"); ta.add_argument("--space", required=True)
    ta.add_argument("--start"); ta.add_argument("--end"); ta.add_argument("--out", default="data/exports")
    return p


def _spaces_by_id(root: Path):
    return {s.space_id: s for s in (load_space(f) for f in discover_space_files(root))}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = load_settings(args.root)
        configure_logging(settings.timezone)
        if args.cmd == "tick":
            files = [Path(s).resolve() for s in args.spaces] or discover_space_files(settings.root)
            if not files:
                raise SbiError("no space files found under config/spaces/")
            outcomes = run_tick(settings, files, use_vlm=not args.no_vlm)
            failed = [o for o in outcomes if not o.ok]
            log.info("tick done: %d ok, %d failed", len(outcomes) - len(failed), len(failed))
            return 1 if failed else 0
        conn = connect(settings.db_path)
        try:
            migrate(conn)
            if args.cmd == "migrate":
                print(f"schema version {conn.execute('PRAGMA user_version').fetchone()[0]}")
            elif args.cmd == "baseline":
                ids = [args.space] if args.space else [r[0] for r in conn.execute("SELECT space_id FROM spaces")]
                for sid in ids:
                    baseline.rebuild(conn, sid, alpha=settings.baseline.ewm_alpha)
                    log.info("baseline rebuilt for %s", sid)
            elif args.cmd == "reprocess":
                space = _spaces_by_id(settings.root).get(args.space)
                if space is None:
                    raise SbiError(f"unknown space {args.space!r}")
                detector = build_detector(settings.detector, settings.root)
                reader = None if args.no_vlm else build_scene_reader(settings.scene_reader)
                ids = args.frame_id or [r[0] for r in conn.execute(
                    "SELECT frame_id FROM frames WHERE space_id=? AND capture_status='ok' ORDER BY ts_utc",
                    (args.space,))]
                done = sum(process_frame(conn, space, i, detector, reader, root=settings.root).processed for i in ids)
                log.info("reprocessed %d/%d frames", done, len(ids))
            elif args.cmd == "eval":
                print(report(conn))
            elif args.cmd == "curate":
                from .curation.curator import curate
                from .curation.services import LmStudioEmbedder, LmStudioJudge
                c, base = settings.curation, settings.scene_reader.base_url
                rep = curate(conn, LmStudioEmbedder(base, c), LmStudioJudge(base, c), min_similarity=c.min_similarity,
                             auto_approve=c.auto_approve and not args.no_auto_approve)
                print(f"names {rep.names} | candidate pairs {rep.candidates} | questions asked {rep.questions_asked}, "
                      f"cached {rep.questions_cached} | new links {len(rep.links)} | re-mapped rows {rep.reresolved_rows}")
                for k in rep.links:
                    print(f"  #{k['link_id']} {k['child']} {k['relation']} {k['parent']} (cos {k['similarity']}, {k['status']})")
            elif args.cmd == "links":
                q = "SELECT link_id, child, relation, parent, status, source, similarity FROM object_links"
                rows = conn.execute(q + (" WHERE status=?" if args.status else "") + " ORDER BY link_id",
                                    (args.status,) if args.status else ()).fetchall()
                for r in rows:
                    print(f"#{r[0]:<4} {r[1]} {r[2]} {r[3]}  [{r[4]}, {r[5]}, cos {r[6]}]")
                print(f"{len(rows)} link(s)")
            elif args.cmd == "link-set":
                from .curation.curator import set_link_status
                try:
                    print(set_link_status(conn, args.link_id, args.status))
                except ValueError as e:
                    raise SbiError(str(e)) from e
            elif args.cmd == "table":
                from .forecast.table import build_table
                space = _spaces_by_id(settings.root).get(args.space)
                if space is None:
                    raise SbiError(f"unknown space {args.space!r}")
                tc = settings.table
                res = build_table(conn, args.space, tz=space.tz, step_minutes=tc.step_minutes, start=args.start,
                                  end=args.end, min_coverage=tc.min_coverage, min_presence=tc.min_presence,
                                  min_valid_slots=tc.min_valid_slots, min_present_slots=tc.min_present_slots,
                                  exclude_labels=tuple(tc.exclude_labels))
                out = settings.resolve(args.out); out.mkdir(parents=True, exist_ok=True)
                res.table.to_csv(out / f"{args.space}_table.csv", index_label="timestamp")
                res.catalog.to_csv(out / f"{args.space}_catalog.csv", index=False)
                st, cat = res.table["slot_status"].value_counts().to_dict(), res.catalog
                print(f"{args.space}: {len(res.table)} slots {res.table.index.min()} .. {res.table.index.max()} IST | {st}")
                print(f"{len(cat)} dimensions, {int(cat['model_ready'].sum())} model-ready | by kind: "
                      f"{cat.groupby('kind')['model_ready'].agg(['size', 'sum']).to_dict('index')}")
                from .forecast.table import select_covariates
                sel = select_covariates(cat, targets=[], max_covariates=tc.max_covariates, always=tc.always_include,
                                        min_autocorr=tc.min_autocorr, autocorr_min_slots=tc.autocorr_min_slots)
                res.catalog.assign(default_input=res.catalog["column"].isin(sel)).to_csv(
                    out / f"{args.space}_catalog.csv", index=False)
                print(f"default model inputs ({len(sel)}): {', '.join(sel)}")
                print(f"written: {out / (args.space + '_table.csv')} and _catalog.csv")
        finally:
            conn.close()
        return 0
    except SbiError as e:
        log.error("%s", e) if logging.getLogger().handlers else print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
