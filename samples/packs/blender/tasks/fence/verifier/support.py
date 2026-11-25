"""Shared, evaluator-owned limits and explicit UTF-8 I/O."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

HERE = Path(__file__).resolve().parent


def load_policy():
    return json.loads((HERE / 'runtime_policy.json').read_text(encoding='utf-8'))


POLICY = load_policy()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON number: ' + value)
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed): raise ValueError('nonfinite JSON number')
        return parsed
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)


def read_json(path, limit=8 * 1024 * 1024):
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('JSON exceeds byte budget')
    return strict_json(data.decode('utf-8'))


def dump_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def file_preflight(path):
    """No hashing, native parsing, or dependency loading before this gate."""
    p = Path(path)
    try:
        info = p.lstat()
    except FileNotFoundError:
        return 'submission_missing'
    if not stat.S_ISREG(info.st_mode):
        return 'submission_not_regular_file'
    if info.st_size > POLICY['max_submission_bytes']:
        return 'submission_size_budget_exceeded'
    if info.st_size < 12:
        return 'submission_too_small'
    descriptor = os.open(p, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(descriptor, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or not _same_file(info, opened):
            return 'submission_changed_during_preflight'
        magic = stream.read(7)
    if not (magic.startswith(b'BLENDER') or magic.startswith(bytes.fromhex('28b52ffd')) or magic.startswith(bytes.fromhex('1f8b'))):
        return 'not_a_blend_file'
    return None


class InvalidSubmission(ValueError):
    pass


class BudgetExceeded(ValueError):
    pass


def native_preflight(bpy):
    """Bound all raw datablocks, including orphan meshes, before depsgraph evaluation."""
    counts = {'objects': len(bpy.data.objects), 'vertices': 0, 'faces': 0, 'nodes': 0}
    if counts['objects'] > POLICY['max_objects']:
        raise BudgetExceeded('object_budget_exceeded')
    for mesh in bpy.data.meshes:
        counts['vertices'] += len(mesh.vertices)
        counts['faces'] += len(mesh.polygons)
        if counts['vertices'] > POLICY['max_vertices'] or counts['faces'] > POLICY['max_faces']:
            raise BudgetExceeded('mesh_budget_exceeded')
    trees = list(bpy.data.node_groups)
    for collection in (bpy.data.materials, bpy.data.worlds, bpy.data.lights, bpy.data.scenes):
        for item in collection:
            for key in ('node_tree', 'compositing_node_group'):
                tree = getattr(item, key, None)
                if tree is not None:
                    trees.append(tree)
    seen = set()
    for tree in trees:
        if tree.as_pointer() in seen:
            continue
        seen.add(tree.as_pointer())
        counts['nodes'] += len(tree.nodes)
        if counts['nodes'] > POLICY['max_nodes']:
            raise BudgetExceeded('node_budget_exceeded')
    return counts


def identity_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


class Deadline:
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds

    def remaining(self):
        value = self.end - time.monotonic()
        if value <= 0:
            raise TimeoutError('total_evaluation_budget_exceeded')
        return value


def _same_file(a, b):
    return (a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns, a.st_ctime_ns) == (b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns, b.st_ctime_ns)


def copy_tree_bounded(source, destination, byte_limit, file_limit, depth_limit=32, readonly=False):
    """Copy using pinned directory FDs on POSIX; reject links, races and devices.

    Windows fallback is for trusted local development only. Production controller
    acceptance requires Linux and the descriptor-based implementation.
    """
    source, destination = Path(source).absolute(), Path(destination).resolve()
    if source.is_symlink() or source.resolve() == destination or source.resolve() in destination.parents:
        raise ValueError('invalid_snapshot_tree')
    destination.mkdir()
    counter = [0, 0]
    def transfer(reader, before, dest):
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError('bundle_link_or_not_regular')
        counter[1] += 1
        if counter[1] > file_limit or counter[0] + before.st_size > byte_limit:
            raise BudgetExceeded('submission_bundle_budget_exceeded')
        with dest.open('xb') as writer:
            written = 0
            while True:
                block = reader.read(1024 * 1024)
                if not block: break
                written += len(block); counter[0] += len(block)
                if written > before.st_size or counter[0] > byte_limit:
                    raise BudgetExceeded('submission_bundle_budget_exceeded')
                writer.write(block)
        if written != before.st_size or not _same_file(before, os.fstat(reader.fileno())):
            raise ValueError('submission_changed_during_copy')
        if readonly: dest.chmod(0o444)
    def visit(fd, target, depth):
        if depth > depth_limit: raise BudgetExceeded('bundle_depth_budget_exceeded')
        initial = os.fstat(fd)
        names = sorted(os.listdir(fd))
        if len(names) > file_limit: raise BudgetExceeded('bundle_entry_budget_exceeded')
        for name in names:
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            dest = target / name
            if stat.S_ISDIR(info.st_mode):
                # O_NOFOLLOW on every component prevents directory-swap escapes.
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    if (info.st_dev, info.st_ino) != (os.fstat(child).st_dev, os.fstat(child).st_ino):
                        raise ValueError('submission_directory_changed')
                    counter[1] += 1
                    if counter[1] > file_limit: raise BudgetExceeded('bundle_entry_budget_exceeded')
                    dest.mkdir(); visit(child, dest, depth + 1)
                finally: os.close(child)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                with os.fdopen(child, 'rb') as reader:
                    before = os.fstat(reader.fileno())
                    if not _same_file(info, before): raise ValueError('submission_changed_during_copy')
                    transfer(reader, before, dest)
            else: raise ValueError('submission_bundle_link_or_not_regular')
        if not _same_file(initial, os.fstat(fd)):
            raise ValueError('submission_directory_changed')
    if os.name == 'posix':
        fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: visit(fd, destination, 0)
        except OSError as exc:
            raise ValueError('submission_tree_changed_or_unreadable') from exc
        finally: os.close(fd)
    else:
        # Windows is not a production controller; reject all reparse points.
        for parent, dirs, names in os.walk(source, followlinks=False):
            rel = Path(parent).relative_to(source)
            if len(rel.parts) > depth_limit: raise BudgetExceeded('bundle_depth_budget_exceeded')
            for name in dirs + names:
                item = Path(parent) / name
                info = item.lstat()
                if item.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('submission_bundle_link')
            for name in dirs:
                counter[1] += 1
                if counter[1] > file_limit: raise BudgetExceeded('bundle_entry_budget_exceeded')
                (destination / rel / name).mkdir()
            for name in names:
                item = Path(parent) / name
                info = item.lstat()
                if not stat.S_ISREG(info.st_mode): raise ValueError('submission_not_regular')
                with item.open('rb') as reader: transfer(reader, info, destination / rel / name)
    return {'bytes': counter[0], 'entries': counter[1]}


def freeze_submission(source, destination):
    source = Path(source).absolute()
    issue = file_preflight(source)
    if issue: raise ValueError(issue)
    copy_tree_bounded(source.parent, destination, POLICY['max_bundle_bytes'], POLICY['max_bundle_files'], readonly=True)
    return Path(destination) / source.name


def run_native(command, log_path, timeout, report_dir):
    """Bound native logs and monitor report growth; deployment still needs a disk quota."""
    import signal
    import subprocess
    import threading
    options = {'start_new_session': True} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **options)
    exceeded = threading.Event()
    errors = []
    def drain():
        count = 0
        try:
            with Path(log_path).open('wb') as log:
                while True:
                    block = process.stdout.read(65536)
                    if not block:
                        break
                    count += len(block)
                    if count > POLICY['max_native_log_bytes']:
                        exceeded.set()
                        break
                    log.write(block)
        except OSError as exc:
            errors.append(exc)
        finally:
            process.stdout.close()
    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    next_scan = 0
    def scan_reports():
        total = count = 0
        for path in Path(report_dir).rglob('*'):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue  # A worker may atomically replace a temporary file.
            count += 1
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
            if total > POLICY['max_report_bytes'] or count > 20000:
                raise BudgetExceeded('report_output_budget_exceeded')
    try:
        while process.poll() is None or thread.is_alive():
            now = time.monotonic()
            if now >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            if now >= next_scan:
                scan_reports()
                next_scan = now + .5
            if exceeded.is_set():
                raise BudgetExceeded('native_log_budget_exceeded')
            time.sleep(.01)
        if exceeded.is_set():
            raise BudgetExceeded('native_log_budget_exceeded')
        if errors:
            raise errors[0]
        scan_reports()  # Include writes made after the last periodic scan.
        return process.returncode
    finally:
        # Kill surviving descendants even when the main process exited normally.
        if os.name == 'posix' or process.poll() is None or thread.is_alive():
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        process.wait(timeout=10)
        thread.join(timeout=2)


def source_identity(root=HERE):
    """Version the actual executable verifier; generated runtime lock is separate."""
    records = {}
    for path in sorted(Path(root).iterdir()):
        if path.is_file() and (path.suffix == '.py' or path.name in ('rubric.json','general-rubric.json','runtime_policy.json','calibration-policy.json')):
            records[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity_hash(records)


def native_file_preflight(path):
    """Stream compressed inputs with bounded output before the native file parser."""
    import gzip
    issue = file_preflight(path)
    if issue: raise InvalidSubmission(issue)
    with Path(path).open('rb') as raw:
        magic = raw.read(7)
    if magic.startswith(b'BLENDER'): return
    errors = (gzip.BadGzipFile, EOFError)
    if magic.startswith(bytes.fromhex('1f8b')):
        opener = gzip.open
    else:
        import zstandard
        opener = zstandard.open
        errors += (zstandard.ZstdError,)
    try:
        with opener(path, 'rb') as stream:
            header = stream.read(12)
            if not header.startswith(b'BLENDER') or len(header) < 12:
                raise InvalidSubmission('compressed_payload_is_not_blend')
            total = len(header)
            while True:
                chunk = stream.read(128 * 1024)
                if not chunk: break
                total += len(chunk)
                if total > POLICY['max_submission_bytes']:
                    raise BudgetExceeded('expanded_submission_size_budget_exceeded')
    except errors as exc:
        raise InvalidSubmission('corrupt_compressed_blend') from exc
