"""Glassbox CLI — command-line tools for inspecting ContextPacks."""

from __future__ import annotations

from typing import Optional


def cli() -> None:
    """Entry point for the glassbox CLI."""
    try:
        import click
    except ImportError:
        print("CLI requires click. Install with: pip install glassbox-ctx[cli]")
        return

    @click.group(invoke_without_command=True)
    @click.option("--version", is_flag=True, help="Show version")
    @click.pass_context
    def main(ctx: click.Context, version: bool) -> None:
        """Glassbox — AI context observability. See what your LLM saw."""
        if version:
            from .format.version import FORMAT_VERSION
            click.echo(f"glassbox v{FORMAT_VERSION}")
            return
        if ctx.invoked_subcommand is None:
            click.echo(ctx.get_help())

    @main.command()
    @click.option("--port", default=4100, help="Port to listen on")
    @click.option("--dir", default=None, help="Path to .glassbox/ directory")
    def serve(port: int, dir: Optional[str]) -> None:
        """Start the Glassbox viewer UI."""
        try:
            from .viewer.server import serve as _serve
            _serve(port=port, directory=dir)
        except ImportError:
            click.echo(
                "Viewer requires server extras. Install with: pip install glassbox-ctx[server]"
            )

    @main.command()
    @click.option("--provider", default="anthropic", type=click.Choice(["anthropic", "openai", "ollama"]),
                  help="LLM provider to proxy")
    @click.option("--target", default=None, help="Custom upstream URL")
    @click.option("--port", "proxy_port", default=4050, help="Proxy port (default 4050)")
    @click.option("--viewer-port", default=4100, help="Viewer port (default 4100)")
    @click.option("--dir", default=None, help="Path to .glassbox/ directory")
    @click.option("--working-dir", default=None, help="Project directory to scan for source inventory")
    def proxy(provider: str, target: Optional[str], proxy_port: int, viewer_port: int, dir: Optional[str], working_dir: Optional[str]) -> None:
        """Start proxy + viewer. Point your LLM client at the proxy port."""
        try:
            from .proxy import proxy as _proxy
            _proxy(provider=provider, target=target, proxy_port=proxy_port,
                   viewer_port=viewer_port, storage_dir=dir, working_dir=working_dir)
        except ImportError:
            click.echo(
                "Proxy requires server extras. Install with: pip install glassbox-ctx[server]"
            )

    @main.command()
    @click.option("--dir", default=None, help="Path to .glassbox/ directory")
    @click.option("--limit", default=20, help="Max runs to show")
    def ls(dir: Optional[str], limit: int) -> None:
        """List recent runs."""
        from .core.file_storage import FileStorage

        storage = FileStorage(dir)
        runs = storage.list_runs(limit=limit)

        if not runs:
            click.echo("No runs found.")
            return

        # Header
        click.echo(
            f"{'Agent':<25} {'Started':<22} {'Steps':>5} {'Tokens':>8} {'Latency':>8} {'Status':<10}"
        )
        click.echo("-" * 85)

        for r in runs:
            name = r.agent_name or r.app_name or "—"
            tokens = r.total_input_tokens + r.total_output_tokens
            latency = f"{r.total_latency_ms / 1000:.1f}s"
            click.echo(
                f"{name:<25} {r.started_at[:19]:<22} {r.step_count:>5} "
                f"{tokens:>8,} {latency:>8} {r.status:<10}"
            )

    @main.command()
    @click.argument("run_id")
    @click.option("--dir", default=None, help="Path to .glassbox/ directory")
    def inspect(run_id: str, dir: Optional[str]) -> None:
        """Inspect a specific run."""
        from .core.file_storage import FileStorage

        storage = FileStorage(dir)
        run = storage.get_run(run_id)

        if not run:
            click.echo(f"Run {run_id} not found.")
            return

        click.echo(f"\nRun: {run.run_id}")
        click.echo(f"Agent: {run.agent_name or '—'}")
        click.echo(f"Status: {run.status}")
        click.echo(f"Steps: {run.step_count}")
        click.echo(f"Tokens: {run.total_input_tokens + run.total_output_tokens:,}")
        click.echo(f"Latency: {run.total_latency_ms / 1000:.1f}s")
        click.echo("")

        steps = storage.get_steps(run_id)
        for step in steps:
            click.echo(
                f"  Step #{step.step_index} [{step.model.model}] "
                f"{step.metrics.input_tokens} in / {step.metrics.output_tokens} out / "
                f"{step.metrics.latency_ms:.0f}ms"
            )
            click.echo(f"    Sections: {len(step.sections)}")
            for s in step.sections:
                click.echo(f"      {s.type} ({s.source or '—'}) — {s.token_count} tokens")

    main()
