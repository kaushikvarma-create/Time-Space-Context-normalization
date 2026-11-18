"""Command line: python -m sbi <command>   (run from the project root with PYTHONPATH=src)

  tick        capture + process every space in config/spaces/ (what cron runs)
  migrate     apply pending database migrations
  baseline    rebuild baselines (all buckets) for one or all spaces
  reprocess   re-run Engine 1 on stored frames (e.g. after a prompt or model change);
              --detector-only re-runs the detector (and segmenter) and keeps each frame's stored VLM answer
  eval        accuracy report against human labels
  curate      find synonym ('same') and hierarchy ('is_a') links between object names, apply them, re-map history
  links       list taxonomy links            link-set ID approved|rejected   human decision (reversible)
  embed       store whole-scene image vectors for frames that have none (history, or frames where it failed)
  forecast    forecast a space with a model (Chronos-2 by default) and store the run in the database
  backtest    forecast from many past moments, score against what happened, compare models and covariate sets
  accuracy    score the forecasts actually made against what was later measured (and against persistence)
  score       score stored photos against the space's probe prompts (history, or a prompt you just added)
  table       build the Chronos-ready table + column catalog for a space (CSV in data/exports/)
  stability   label-free noise check: how steady each count is inside a quiet window (e.g. night)

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
from .perception.factory import build_detector, build_embedder, build_scene_reader, build_scorer, build_segmenter
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
    r.add_argument("--detector-only", action="store_true", help="keep the stored VLM answer; no VLM call")
    sub.add_parser("eval")
    cu = sub.add_parser("curate"); cu.add_argument("--no-auto-approve", action="store_true")
    li = sub.add_parser("links"); li.add_argument("--status", choices=["proposed", "approved", "rejected"])
    ls = sub.add_parser("link-set"); ls.add_argument("link_id", type=int)
    ls.add_argument("status", choices=["approved", "rejected"])
    ta = sub.add_parser("table"); ta.add_argument("--space", required=True)
    ta.add_argument("--start"); ta.add_argument("--end"); ta.add_argument("--out", default="data/exports")
    ta.add_argument("--fit-end", help="learn the image-vector components from slots before this local time only "
                                      "(keeps a back-test's future out of the fit)")
    fc = sub.add_parser("forecast"); fc.add_argument("--space", required=True)
    fc.add_argument("--variant", help="covariate set (default: forecast.variant in config/pipeline.yaml)")
    fc.add_argument("--horizon", type=int, help="slots ahead (default: forecast.horizon)")
    fc.add_argument("--model", default="chronos2", help="chronos2 | seasonal_mean | seasonal_naive | persistence")
    bt = sub.add_parser("backtest"); bt.add_argument("--space", required=True)
    bt.add_argument("--horizon", type=int); bt.add_argument("--min-context", type=int)
    bt.add_argument("--step", type=int, default=8, help="slots between forecast origins (default 8 = 2 hours)")
    bt.add_argument("--variants", help="comma-separated covariate sets (default: all in forecast.variants)")
    bt.add_argument("--models", default="chronos2,seasonal_mean,seasonal_naive,persistence")
    bt.add_argument("--out", default="data/exports")
    ac = sub.add_parser("accuracy"); ac.add_argument("--space", required=True)
    ac.add_argument("--since", help="only runs made from this local time on, e.g. '2026-11-17'")
    ac.add_argument("--out", default="data/exports")
    sc = sub.add_parser("score"); sc.add_argument("--space", required=True)
    sc.add_argument("--limit", type=int, help="stop after this many frames")
    em = sub.add_parser("embed"); em.add_argument("--space", required=True)
    em.add_argument("--limit", type=int, help="stop after this many frames")
    sb = sub.add_parser("stability"); sb.add_argument("--space", required=True)
    sb.add_argument("--start", help="quiet window start, local time, e.g. '2026-11-14 00:00'")
    sb.add_argument("--end", help="quiet window end, local time")
    sb.add_argument("--min-mean", type=float, default=0.0, help="hide columns whose window mean is below this")
    sb.add_argument("--out", default="data/exports")
    return p


def _spaces_by_id(root: Path):
    return {s.space_id: s for s in (load_space(f) for f in discover_space_files(root))}


def _build_table(conn, settings, space, start=None, end=None, fit_end=None, max_slots=None):
    from .forecast.build import table_for
    return table_for(conn, settings, space, start=start, end=end, fit_end=fit_end, max_slots=max_slots)


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
                segmenter = build_segmenter(settings.segmenter, settings.root)    # SegmenterError: stop, do not
                                                                                  # silently wipe stored outlines
                if args.detector_only and args.no_vlm:
                    raise SbiError("--detector-only keeps the stored VLM answers; it cannot be combined with --no-vlm")
                reader = None if args.no_vlm else build_scene_reader(settings.scene_reader)
                ids = args.frame_id or [r[0] for r in conn.execute(
                    "SELECT frame_id FROM frames WHERE space_id=? AND capture_status='ok' ORDER BY ts_utc",
                    (args.space,))]
                done, failed = 0, 0
                for i in ids:
                    try:
                        done += process_frame(conn, space, i, detector, reader, root=settings.root,
                                              replay_vlm=args.detector_only, segmenter=segmenter,
                                              dedupe_outline_iou=settings.detector.dedupe_outline_iou).processed
                    except SbiError as e:      # that frame is left exactly as it was; keep going
                        failed += 1
                        log.error("frame %d not reprocessed: %s", i, e)
                log.info("reprocessed %d/%d frames%s", done, len(ids), f", {failed} failed" if failed else "")
                if failed:
                    return 1
            elif args.cmd == "embed":
                from .embedding import embed_missing
                if args.space not in _spaces_by_id(settings.root):
                    raise SbiError(f"unknown space {args.space!r}")
                embedder = build_embedder(settings.embedder, settings.root)
                if embedder is None:
                    raise SbiError("embedder.kind is none in config/pipeline.yaml: nothing to embed with")
                rep = embed_missing(conn, args.space, embedder, root=settings.root, limit=args.limit)
                log.info("%s: embedded %d frames with %s%s", args.space, rep.done, embedder.version,
                         f", {rep.failed} failed" if rep.failed else "")
                if rep.failed:
                    return 1
            elif args.cmd == "forecast":
                from .forecast.models import build_forecaster
                from .forecast.runner import forecast_space
                space = _spaces_by_id(settings.root).get(args.space)
                if space is None:
                    raise SbiError(f"unknown space {args.space!r}")
                fcs = settings.forecast
                forecaster = build_forecaster(args.model, model=fcs.model, cross_learning=fcs.cross_learning)
                run = forecast_space(conn, space, _build_table(conn, settings, space, max_slots=fcs.max_context), forecaster, settings=fcs,
                                     table_settings=settings.table, variant=args.variant, horizon=args.horizon,
                                     step_minutes=settings.table.step_minutes)
                print(f"{args.space}: run {run.run_id} | {forecaster.name} | origin {run.origin} IST | "
                      f"{len(run.targets)} targets, {len(run.covariates)} past covariates")
                shown = run.forecast[run.forecast["step"] <= 4]
                for tgt in run.targets[:6]:
                    rows = shown[shown["target"] == tgt]
                    print(f"  {tgt}: " + " | ".join(
                        f"{r.timestamp:%H:%M} " + "/".join(f"{getattr(r, c):.2f}" for c in run.forecast.columns
                                                          if c.startswith("q")) for r in rows.itertuples()))
                print(f"stored in forecast_runs / forecasts (run_id {run.run_id}); quantiles in column order: "
                      f"{', '.join(c for c in run.forecast.columns if c.startswith('q'))}")
            elif args.cmd == "backtest":
                from .forecast.backtest import BacktestPlan, origins_for, run_backtest
                from .forecast.metrics import overall, summarise
                from .forecast.models import build_forecaster
                space = _spaces_by_id(settings.root).get(args.space)
                if space is None:
                    raise SbiError(f"unknown space {args.space!r}")
                fcs, tc = settings.forecast, settings.table
                names = [v.strip() for v in args.variants.split(",")] if args.variants else list(fcs.variants)
                if any(v not in fcs.variants for v in names):
                    raise SbiError(f"unknown variant in {names}; choose from {sorted(fcs.variants)}")
                whole_res = _build_table(conn, settings, space)
                whole = whole_res.table
                plan = BacktestPlan(
                    horizon=args.horizon or fcs.horizon, step=args.step, min_context=args.min_context or fcs.min_context,
                    quantiles=fcs.quantiles, variants={v: fcs.variants[v] for v in names},
                    patterns=space.forecast_targets, cal=whole_res.cal,
                    catalog_args=dict(min_coverage=tc.min_coverage, min_presence=tc.min_presence,
                                      min_valid_slots=tc.min_valid_slots, min_present_slots=tc.min_present_slots,
                                      exclude_labels=tuple(tc.exclude_labels)))
                first = origins_for(len(whole), plan)
                if not first:
                    raise SbiError(f"not enough history to backtest {args.space}: need at least "
                                   f"{plan.min_context + plan.horizon} slots")
                # image-vector components are learned from the history before the first origin only (no look-ahead)
                res = _build_table(conn, settings, space, fit_end=str(whole.index[first[0]]))
                forecasters = [build_forecaster(m.strip(), model=fcs.model, cross_learning=fcs.cross_learning)
                               for m in args.models.split(",")]
                bt = run_backtest(res.table, args.space, forecasters, plan, settings=fcs, table_settings=tc)
                summary = summarise(bt.raw)
                out = settings.resolve(args.out); out.mkdir(parents=True, exist_ok=True)
                bt.raw.to_csv(out / f"{args.space}_backtest_raw.csv", index=False)
                summary.to_csv(out / f"{args.space}_backtest.csv", index=False)
                print(f"{args.space}: {bt.origins} origins every {plan.step} slots, {plan.horizon} slots ahead, "
                      f"{plan.min_context}+ slots of history, {len(summary['target'].unique()) if len(summary) else 0} targets")
                if bt.failures:
                    print(f"skipped origins: {bt.failures}")
                print("skill = 1 - MAE / MAE of seasonal_mean on the same points (positive beats the weekday-by-time "
                      "average); wql lower is better; coverage should be near the interval's nominal level")
                print(overall(summary).round(3).to_string(index=False))
                print(f"written: {out / (args.space + '_backtest.csv')} and _backtest_raw.csv")
            elif args.cmd == "accuracy":
                from .forecast.accuracy import by_lead, by_target, live_points
                space = _spaces_by_id(settings.root).get(args.space)
                if space is None:
                    raise SbiError(f"unknown space {args.space!r}")
                points = live_points(conn, args.space, _build_table(conn, settings, space).table, since=args.since)
                if points.empty:
                    print(f"{args.space}: no stored forecast has a measured outcome yet (forecasts are scored once "
                          f"their time has passed)")
                else:
                    out = settings.resolve(args.out); out.mkdir(parents=True, exist_ok=True)
                    by_target(points).to_csv(out / f"{args.space}_accuracy.csv", index=False)
                    print(f"{args.space}: {points['run_id'].nunique()} forecast runs, {len(points)} scored points | "
                          f"skill = 1 - MAE / MAE of holding the last value (positive is better)")
                    print(by_lead(points).round(3).to_string(index=False))
                    print(f"written: {out / (args.space + '_accuracy.csv')} (per target)")
            elif args.cmd == "score":
                from .scoring import score_missing
                space = _spaces_by_id(settings.root).get(args.space)
                if space is None:
                    raise SbiError(f"unknown space {args.space!r}")
                if not space.probes:
                    raise SbiError(f"{args.space} has no `probes` in its space file: nothing to score")
                scorer = build_scorer(settings.scorer, settings.root)
                if scorer is None:
                    raise SbiError("scorer.kind is none in config/pipeline.yaml: nothing to score with")
                rep = score_missing(conn, space, scorer, root=settings.root, limit=args.limit)
                log.info("%s: scored %d frames with %s%s", args.space, rep.done, scorer.version,
                         f", {rep.failed} failed" if rep.failed else "")
                if rep.failed:
                    return 1
            elif args.cmd == "eval":
                print(report(conn, mask_dup_iou=settings.table.mask_dup_iou))
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
                space = _spaces_by_id(settings.root).get(args.space)
                if space is None:
                    raise SbiError(f"unknown space {args.space!r}")
                tc = settings.table
                res = _build_table(conn, settings, space, args.start, args.end, args.fit_end)
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
            elif args.cmd == "stability":
                from .eval.stability import stability
                space = _spaces_by_id(settings.root).get(args.space)
                if space is None:
                    raise SbiError(f"unknown space {args.space!r}")
                rep = stability(_build_table(conn, settings, space).table, start=args.start, end=args.end)
                out = settings.resolve(args.out); out.mkdir(parents=True, exist_ok=True)
                rep.to_csv(out / f"{args.space}_stability.csv", index=False)
                shown = rep[rep["window_mean"].abs() >= args.min_mean].drop(columns=["kind", "source"])
                print(f"{args.space}: quiet window {args.start or 'start'} .. {args.end or 'end'} ({space.tz}); "
                      f"noise = mean change between consecutive slots, snr = spread / noise (higher is better)")
                print(shown.to_string(index=False) if len(shown) else "no columns with data in that window")
                print(f"written: {out / (args.space + '_stability.csv')}")
        finally:
            conn.close()
        return 0
    except SbiError as e:
        log.error("%s", e) if logging.getLogger().handlers else print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
