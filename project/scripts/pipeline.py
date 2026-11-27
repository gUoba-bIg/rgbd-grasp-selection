#!/usr/bin/env python
"""Run independently verifiable experiment stages with recoverable state."""
from __future__ import print_function

import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

sys.dont_write_bytecode = True


class PipelineError(Exception):
    pass


def timestamp():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def digest_json(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise PipelineError("Project paths must be nonempty relative paths: {!r}".format(relative))
    path = root / relative
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PipelineError("Path leaves the project directory: " + relative)
    if resolved == root:
        raise PipelineError("The project root cannot be an artifact path.")
    return path


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def artifact_digest(root, relative):
    path = inside(root, relative)
    if not path.exists():
        raise PipelineError("Required artifact is missing: " + relative)
    if path.is_symlink():
        raise PipelineError("Artifact links are not supported: " + relative)
    if path.is_file():
        return {"kind": "file", "sha256": file_digest(path), "bytes": path.stat().st_size}
    if path.is_dir():
        entries = []
        for item in sorted(path.rglob("*"), key=lambda entry: entry.as_posix()):
            inside(root, item.relative_to(root).as_posix())
            if item.is_symlink():
                raise PipelineError("Artifact links are not supported: " + str(item))
            name = item.relative_to(path).as_posix()
            if item.is_file():
                entries.append([name, "file", item.stat().st_size, file_digest(item)])
            elif item.is_dir():
                entries.append([name, "directory"])
            else:
                raise PipelineError("Unsupported artifact: " + str(item))
        return {"kind": "directory", "sha256": digest_json(entries), "entries": len(entries)}
    raise PipelineError("Unsupported artifact: " + relative)


@contextlib.contextmanager
def project_lock(root):
    path = root / "runtime" / "pipeline" / "execution.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    acquired = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise PipelineError("Another stage runner holds the project lock. Wait for it to finish.")
        acquired = True
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            # Access denied is inconclusive; prevent a second writer conservatively.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def project_environment(root):
    env = dict(os.environ)
    cache = root / "runtime" / "cache"
    paths = {
        "XDG_CACHE_HOME": cache,
        "TORCH_HOME": cache / "torch",
        "HF_HOME": cache / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "transformers",
        "PIP_CACHE_DIR": cache / "pip",
        "PYTHONPYCACHEPREFIX": cache / "pycache",
        "MPLCONFIGDIR": cache / "matplotlib",
        "CUDA_CACHE_PATH": cache / "cuda",
        "TMP": root / "runtime" / "tmp",
        "TEMP": root / "runtime" / "tmp",
        "TMPDIR": root / "runtime" / "tmp",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env["PROJECT_ROOT"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # Python 3.7 has no pycache prefix support.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class Pipeline:
    def __init__(self, root, config="configs/stages.json"):
        self.root = Path(root).resolve()
        self.state_path = self.root / "runtime" / "pipeline" / "state.json"
        self.previous_path = self.state_path.with_name("state.previous.json")
        try:
            with inside(self.root, config).open(encoding="utf-8-sig") as stream:
                raw = json.load(stream)
        except (OSError, ValueError) as error:
            raise PipelineError("Cannot read stage configuration: " + str(error))
        self.stages = {}
        for stage in raw.get("stages", []):
            identifier = stage.get("id", "")
            if not re.match(r"^[A-Za-z0-9_-]+$", identifier) or identifier in self.stages:
                raise PipelineError("Invalid or duplicate stage id: " + str(identifier))
            if not isinstance(stage.get("implemented"), bool):
                raise PipelineError(identifier + ": implemented must be true or false.")
            for key in ("depends_on", "outputs", "inputs"):
                values = stage.get(key, [])
                if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                    raise PipelineError(identifier + ": " + key + " must be a list of strings.")
            if len(set(stage.get("depends_on", []))) != len(stage.get("depends_on", [])):
                raise PipelineError(identifier + ": duplicate dependencies.")
            if stage["implemented"]:
                command = stage.get("command")
                if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                    raise PipelineError(identifier + ": an implemented stage needs a command list.")
                if not stage.get("outputs"):
                    raise PipelineError(identifier + ": an implemented stage needs required outputs.")
            for name in stage.get("outputs", []) + stage.get("inputs", []):
                inside(self.root, name)
            self.stages[identifier] = stage
        if not self.stages:
            raise PipelineError("No stages are configured.")
        owners = []
        for identifier, stage in self.stages.items():
            for dependency in stage.get("depends_on", []):
                if dependency not in self.stages:
                    raise PipelineError(identifier + ": unknown dependency " + dependency)
            for name in stage.get("outputs", []):
                path = inside(self.root, name).resolve()
                for previous, owner in owners:
                    if path == previous or path in previous.parents or previous in path.parents:
                        raise PipelineError("Overlapping outputs: " + owner + " and " + identifier)
                owners.append((path, identifier))
        for identifier in self.stages:
            self.order_through(identifier)
        self.state = self.load_state()

    def load_state(self):
        if not self.state_path.exists() and not self.previous_path.exists():
            return {"schema_version": 1, "stages": {}}
        for path in (self.state_path, self.previous_path):
            try:
                with path.open(encoding="utf-8") as stream:
                    state = json.load(stream)
                if state.get("schema_version") != 1 or not isinstance(state.get("stages"), dict):
                    raise ValueError("unsupported state format")
                if path == self.previous_path:
                    print("Recovering the previous valid state snapshot.", file=sys.stderr)
                return state
            except (OSError, ValueError, AttributeError):
                pass
        raise PipelineError("Both state snapshots are unreadable; inspect runtime/pipeline before continuing.")

    def save_state(self):
        if self.state_path.exists():
            try:
                with self.state_path.open(encoding="utf-8") as stream:
                    previous = json.load(stream)
                if previous.get("schema_version") == 1 and isinstance(previous.get("stages"), dict):
                    atomic_json(self.previous_path, previous)
            except (OSError, ValueError, AttributeError):
                pass
        self.state["updated_at"] = timestamp()
        atomic_json(self.state_path, self.state)

    def order_through(self, identifier):
        if identifier not in self.stages:
            raise PipelineError("Unknown stage: " + identifier)
        ordered, visiting, seen = [], set(), set()

        def visit(item):
            if item in visiting:
                raise PipelineError("Dependency cycle contains " + item)
            if item in seen:
                return
            visiting.add(item)
            for dependency in self.stages[item].get("depends_on", []):
                visit(dependency)
            visiting.remove(item)
            seen.add(item)
            ordered.append(item)

        visit(identifier)
        return ordered

    def command(self, stage):
        return [part.replace("{python}", sys.executable).replace("{root}", str(self.root))
                for part in stage.get("command", [])]

    def signature(self, identifier, memo=None):
        stage = self.stages[identifier]
        dependencies = {}
        for dependency in stage.get("depends_on", []):
            dependencies[dependency] = self.verify(dependency, memo)
        inputs = {name: artifact_digest(self.root, name) for name in stage.get("inputs", [])}
        # Script changes invalidate a completed run even without explicit inputs.
        for part in stage.get("command", []):
            if part.endswith((".py", ".ps1", ".sh")) and "{" not in part and not Path(part).is_absolute():
                candidate = inside(self.root, part)
                if candidate.is_file():
                    inputs[part] = artifact_digest(self.root, part)
        content = {"stage": stage, "command": self.command(stage), "inputs": inputs,
                   "dependencies": dependencies}
        return {"input_fingerprint": digest_json(content), "dependencies": dependencies,
                "stage_config_sha256": digest_json(stage), "inputs": inputs}

    def verify(self, identifier, memo=None):
        memo = {} if memo is None else memo
        if identifier in memo:
            return memo[identifier]
        record = self.state["stages"].get(identifier)
        if not record or record.get("status") != "complete":
            raise PipelineError("Dependency " + identifier + " has not completed successfully.")
        if not self.stages[identifier]["implemented"]:
            raise PipelineError(identifier + " is currently marked as planned.")
        signature = self.signature(identifier, memo)
        if signature["input_fingerprint"] != record.get("input_fingerprint"):
            raise PipelineError(identifier + ": configuration, code, inputs or dependency fingerprint changed; use --new-run explicitly.")
        expected = record.get("outputs", {})
        try:
            actual = {name: artifact_digest(self.root, name) for name in self.stages[identifier]["outputs"]}
        except PipelineError as error:
            raise PipelineError(identifier + ": " + str(error) + "; use --new-run explicitly.")
        if actual != expected:
            raise PipelineError(identifier + ": output hashes changed; use --new-run explicitly.")
        fingerprint = digest_json({"input_fingerprint": signature["input_fingerprint"], "outputs": actual})
        memo[identifier] = fingerprint
        return fingerprint

    def archive(self, identifier, record):
        destination = self.root / "runtime" / "pipeline" / "archive" / identifier / (datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
        destination.mkdir(parents=True)
        atomic_json(destination / "record.json", record or {})
        names = set(self.stages[identifier].get("outputs", []))
        names.update((record or {}).get("outputs", {}))
        for name in sorted(names):
            source = inside(self.root, name)
            if not source.exists():
                continue
            artifact_digest(self.root, name)
            target = destination / "artifacts" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(str(source), str(target))
            else:
                shutil.copy2(str(source), str(target))
        return destination.relative_to(self.root).as_posix()

    def run_stage(self, identifier, resume=False, new_run=False):
        stage = self.stages[identifier]
        if not stage["implemented"]:
            print("{} PLANNED: {}".format(identifier, stage.get("title", identifier)))
            raise PipelineError(identifier + " has no implemented command; no completion is recorded.")
        signature = self.signature(identifier)
        previous = self.state["stages"].get(identifier)
        archive = None
        if previous:
            if previous.get("status") == "running" and pid_alive(previous.get("child_pid")):
                raise PipelineError(identifier + ": the previous child process may still be active; verify it has exited before restarting.")
            if previous.get("status") == "complete" and not new_run:
                self.verify(identifier)
                print(identifier + " SKIP: verified complete.")
                return
            if previous.get("input_fingerprint") != signature["input_fingerprint"] and not new_run:
                raise PipelineError(identifier + ": configuration, code, inputs or dependencies changed; use --new-run explicitly.")
            if not new_run and not resume:
                raise PipelineError(identifier + ": unfinished attempt exists; use --resume to continue or --new-run explicitly.")
        elif not new_run and any(inside(self.root, name).exists() for name in stage["outputs"]):
            raise PipelineError(identifier + ": existing outputs have no stage record; use --new-run explicitly to preserve and rerun them.")
        if new_run:
            archive = self.archive(identifier, previous)
        attempt = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        log_dir = self.root / "logs" / "pipeline" / identifier / attempt
        log_dir.mkdir(parents=True)
        record = dict(signature)
        record.update({"status": "running", "attempt_id": attempt, "started_at": timestamp(),
                       "command": self.command(stage), "parent_pid": os.getpid(),
                       "stdout": (log_dir / "stdout.log").relative_to(self.root).as_posix(),
                       "stderr": (log_dir / "stderr.log").relative_to(self.root).as_posix(),
                       "resume": bool(resume and previous and not new_run), "archive": archive})
        self.state["stages"][identifier] = record
        self.save_state()
        env = project_environment(self.root)
        env["PIPELINE_STAGE_ID"] = identifier
        env["PIPELINE_ATTEMPT_ID"] = attempt
        env["PIPELINE_RESUME"] = "1" if record["resume"] else "0"
        env["PIPELINE_NEW_RUN"] = "1" if new_run else "0"
        child = None
        print(identifier + " RUN: " + str(log_dir), flush=True)
        try:
            with (log_dir / "stdout.log").open("ab", buffering=0) as out, (log_dir / "stderr.log").open("ab", buffering=0) as err:
                child = subprocess.Popen(record["command"], cwd=str(self.root), env=env,
                                         stdout=out, stderr=err, shell=False, close_fds=True)
                record["child_pid"] = child.pid
                self.save_state()
                code = child.wait()
                os.fsync(out.fileno())
                os.fsync(err.fileno())
            record["returncode"] = code
            if code != 0:
                raise PipelineError(identifier + ": command returned " + str(code) + "; inspect the stage logs.")
            record["outputs"] = {name: artifact_digest(self.root, name) for name in stage["outputs"]}
            decisions = {}
            for name in stage["outputs"]:
                if name.endswith(".json") and inside(self.root, name).is_file():
                    try:
                        with inside(self.root, name).open(encoding="utf-8-sig") as stream:
                            data = json.load(stream)
                        if isinstance(data, dict) and "decision" in data:
                            decisions[name] = data["decision"]
                    except (OSError, ValueError):
                        pass
            record["decisions"] = decisions
            record["status"] = "complete"
            record["completed_at"] = timestamp()
            self.save_state()
            print(identifier + " COMPLETE: outputs verified.")
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            record["error"] = str(error) or type(error).__name__
            record["ended_at"] = timestamp()
            self.save_state()
            raise

    def status(self, active=False):
        rows = []
        for identifier, stage in self.stages.items():
            record = self.state["stages"].get(identifier, {})
            state = record.get("status", "pending") if stage["implemented"] else "planned"
            reason = ""
            if state == "complete":
                try:
                    self.verify(identifier)
                except PipelineError as error:
                    state, reason = "invalid", str(error)
            elif state == "running" and not active:
                if pid_alive(record.get("child_pid")):
                    state, reason = "child_active", "The runner ended but its child process may still be active."
                else:
                    state, reason = "interrupted", "Use --resume to restart the command from its own saved progress."
            rows.append({"id": identifier, "title": stage.get("title", identifier), "status": state,
                         "reason": reason, "decisions": record.get("decisions", {})})
        return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run recoverable experiment stages.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default="configs/stages.json")
    commands = parser.add_subparsers(dest="action")
    status = commands.add_parser("status")
    status.add_argument("--json", action="store_true")
    run = commands.add_parser("run")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stage")
    selection.add_argument("--through")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--new-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.action:
        parser.print_help()
        return 2
    try:
        root = Path(args.root).resolve()
        if args.action == "status":
            active = False
            try:
                with project_lock(root):
                    pass
            except PipelineError:
                active = True
            pipeline = Pipeline(root, args.config)
            rows = pipeline.status(active)
            if args.json:
                print(json.dumps({"active": active, "stages": rows}, ensure_ascii=False, indent=2))
            else:
                for row in rows:
                    print("{id}: {status} | {title}".format(**row))
                    if row["reason"]:
                        print("  " + row["reason"])
            return 0
        with project_lock(root):
            pipeline = Pipeline(root, args.config)
            target = args.stage or args.through
            order = pipeline.order_through(target)
            selected = [target] if args.stage else order
            for identifier in selected:
                pipeline.run_stage(identifier, args.resume, args.new_run)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Run the same stage with --resume to continue.", file=sys.stderr)
        return 130
    except (PipelineError, OSError, ValueError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
