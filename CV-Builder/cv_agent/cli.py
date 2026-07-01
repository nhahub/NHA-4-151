"""
cv_agent.cli — CLI entry point for CV generation and API server.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from cv_agent.cache import get_cache
from cv_agent.config import PipelineConfig, _FASTAPI_AVAILABLE
from cv_agent.file_parsing import parse_resume_file
from cv_agent.pipeline import run_pipeline
from cv_agent.schemas import UserProfile

def _build_cli():
    p = argparse.ArgumentParser(prog="cv_agent", description="Agentic CV Builder v7.0",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Generate a CV")
    for a, kw in [("--name",{"required":True}),("--role",{"required":True}),
        ("--industry",{"default":""}),("--years",{"default":""}),
        ("--tone",{"default":"professional","choices":["professional","creative","technical"]}),
        ("--email",{"required":True,"help":"Candidate Email"}),
        ("--phone",{"required":True,"help":"Candidate Phone Number"}),
        ("--linkedin",{"required":True,"help":"Candidate LinkedIn URL"}),
        ("--skills",{"required":True,"help":"Comma-separated"}),
        ("--experience",{"required":True,"help":"Semicolon-separated"}),
        ("--education",{"default":""}),("--achievements",{"default":""}),
        ("--certifications",{"default":""}),("--summary",{"default":""}),
        ("--jd",{"default":""}),("--resume",{"default":""}),
        ("--max-iterations",{"type":int,"default":None}),
        ("--threshold",{"type":int,"default":None}),
        ("--candidates",{"type":int,"default":None}),
        ("--session-id",{"default":""}),("--out-md",{"default":""}),
        ("--out-pdf",{"default":""}),("--out-json",{"default":""})]:
        r.add_argument(a, **kw)
    if _FASTAPI_AVAILABLE:
        s = sub.add_parser("serve", help="Start FastAPI server")
        s.add_argument("--host", default="0.0.0.0")
        s.add_argument("--port", type=int, default=8000)
        s.add_argument("--workers", type=int, default=1)
        s.add_argument("--reload", action="store_true")
    return p

def _cmd_run(args):
    profile = UserProfile(
        full_name=args.name, target_role=args.role,
        target_industry=args.industry, years_experience=args.years,
        tone=args.tone, summary=args.summary,
        email=args.email, phone=args.phone, linkedin=args.linkedin,
        skills=[s.strip() for s in args.skills.split(",") if s.strip()],
        experiences=[e.strip() for e in args.experience.split(";") if e.strip()],
        education=[e.strip() for e in args.education.split(";") if e.strip()],
        achievements=[a.strip() for a in args.achievements.split(";") if a.strip()],
        certifications=[c.strip() for c in args.certifications.split(",") if c.strip()],
    )
    if not profile.is_complete():
        print("ERROR: --name, --role, --skills, and --experience are required.")
        raise SystemExit(1)
    jd_text = ""
    if args.jd:
        p = Path(args.jd)
        jd_text = p.read_text(encoding="utf-8") if p.exists() else args.jd
    parsed_resume = ""
    if args.resume:
        try:
            parsed_resume = parse_resume_file(args.resume)
            print(f"[resume] Parsed {len(parsed_resume)} chars from {args.resume}")
        except Exception as e:
            print(f"[resume] WARNING: Could not parse resume — {e}")
    cfg = PipelineConfig()
    if args.max_iterations is not None: cfg.max_iterations = args.max_iterations
    if args.threshold is not None: cfg.score_threshold = args.threshold
    if args.candidates is not None: cfg.num_candidates = args.candidates
    result = run_pipeline(profile=profile, job_description=jd_text,
        parsed_resume=parsed_resume, config=cfg,
        session_id=args.session_id, status_callback=print)
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Session      : {result.session_id}")
    print(f"  Candidate    : {result.candidate_name}")
    print(f"  Role         : {result.target_role}")
    print(f"  Iterations   : {result.total_iterations}")
    print(f"  Latency (ms) : {result.total_latency_ms}")
    if result.final_scores:
        s = result.final_scores
        print(f"  Scores       : Overall={s.overall_score}  Clarity={s.clarity_score}  "
              f"Structure={s.structure_score}  Impact={s.impact_score}  "
              f"Skills={s.skills_relevance_score}  ATS={s.ats_readiness_score}")
    print(f"  Cache stats  : {get_cache(cfg).stats()}")
    if result.node_errors:
        print(f"  Warnings     : {len(result.node_errors)} node error(s)")
    print(f"{sep}\n")
    print(result.final_cv)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(result.final_cv, encoding="utf-8")
        print(f"\n[out] Markdown → {args.out_md}")
    if args.out_pdf:
        try:
            from cv_agent.pdf_export import export_pdf_file
            export_pdf_file(result.final_cv, result.candidate_name, args.out_pdf)
            print(f"[out] PDF      → {args.out_pdf}")
        except ImportError as e:
            print(f"[out] PDF export skipped: {e}")
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result.to_report_dict(), indent=2), encoding="utf-8")
        print(f"[out] JSON     → {args.out_json}")

def _cmd_serve(args):
    if not _FASTAPI_AVAILABLE:
        print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn")
        raise SystemExit(1)
    import uvicorn
    uvicorn.run("cv_agent.api:app", host=args.host, port=args.port,
                workers=args.workers, reload=args.reload)

def main():
    args = _build_cli().parse_args()
    if args.command == "run": _cmd_run(args)
    elif args.command == "serve": _cmd_serve(args)

if __name__ == "__main__":
    main()
