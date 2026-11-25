"""Controller-owned worker lifecycle; untrusted native workers only write staging.

No container receives a writable alias of inputs, prior evidence or final rewards.
"""
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
from runtime import sandbox
from support import POLICY, BudgetExceeded, copy_tree_bounded, dump_json, run_native


class WorkerRuntime:
    def __init__(self, args, source, output, verifier, deadline):
        self.args, self.source, self.output, self.verifier, self.deadline = args, Path(source), Path(output), Path(verifier), deadline
        self.stages = {}
        self.events = []

    def command(self, script, options, tag, python=False):
        if not re.fullmatch('[A-Za-z0-9_-]{1,64}', tag):
            raise ValueError('invalid worker tag')
        script, options = Path(script), list(options)
        if self.args.backend == 'local':
            if python: return [sys.executable, str(script), *options]
            return [self.args.blender, '--background', '--factory-startup', '--disable-autoexec',
                    '--python-exit-code', '3', '--python', str(script), '--', *options]
        if script.parent.resolve() != self.verifier.resolve(): raise ValueError('untrusted worker script')
        key = '--report-dir' if python else '--output'
        index = options.index(key) + 1
        target = PurePosixPath(options[index])
        if not target.is_relative_to('/output') or '..' in target.parts:
            raise ValueError('worker output must be inside report directory')
        final = self.output.joinpath(*target.relative_to('/output').parts)
        if final == self.output or final.exists(): raise ValueError('worker output must be fresh')
        stage = self.output / '.worker-staging' / tag
        stage.mkdir(parents=True, exist_ok=False)
        self.stages[tag] = (stage, final)
        options[index] = '/scratch/result'
        cmd = sandbox(self.args.image)
        cmd.remove('--rm')  # Preserve exit/OOM state for inspection, then explicitly clean up.
        cmd += ['--cidfile', str(self.output / (tag + '.cid')),
                '--mount', f'type=bind,src={self.source.parent},dst=/input,readonly',
                '--mount', f'type=bind,src={self.verifier},dst=/verifier,readonly',
                '--mount', f'type=bind,src={self.output},dst=/output,readonly',
                '--mount', f'type=bind,src={stage},dst=/scratch', self.args.image]
        if python: return cmd + ['python3', '/verifier/' + script.name, *options]
        return cmd + ['blender', '--background', '--factory-startup', '--disable-autoexec',
                      '--python-exit-code', '3', '--python', '/verifier/' + script.name, '--', *options]

    def run(self, command, tag):
        started = time.monotonic()
        event = {'tag': tag, 'backend': self.args.backend, 'exit_code': None, 'reason': None}
        code = -2
        try:
            code = run_native(command, self.output / (tag + '.log'), self.deadline.remaining(), self.output)
            event['exit_code'] = code
        except (subprocess.TimeoutExpired, TimeoutError):
            code = -1; event['reason'] = 'evaluation_deadline_exceeded'
        except BudgetExceeded as exc:
            code = -3; event['reason'] = str(exc)
        except OSError as exc:
            event['reason'] = type(exc).__name__
        finally:
            cid_path = self.output / (tag + '.cid')
            if self.args.backend == 'docker' and cid_path.is_file():
                cid = cid_path.read_text(encoding='utf-8').strip()
                if not re.fullmatch('[a-f0-9]{12,64}', cid):
                    event['reason'] = 'invalid_container_id'; code = -2
                else:
                    try:
                        data = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', cid],
                                              capture_output=True, encoding='utf-8', timeout=10, check=True)
                        state = json.loads(data.stdout)
                        event['container'] = {k: state.get(k) for k in ('ExitCode', 'OOMKilled', 'Error', 'Status')}
                        if state.get('OOMKilled'):
                            event['reason'] = 'worker_oom_needs_triage'; code = -4
                        elif code in (0, 2) and (state.get('Status') != 'exited' or state.get('ExitCode') != code or state.get('Error')):
                            event['reason'] = 'worker_exit_state_mismatch'; code = -2
                    except (OSError, ValueError, subprocess.SubprocessError):
                        event['inspection_error'] = True
                        if code == 0: code = -2
                    finally:
                        try:
                            subprocess.run(['docker', 'rm', '-f', cid], stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, timeout=15, check=True)
                            event['container_removed'] = True
                        except (OSError, subprocess.SubprocessError):
                            event['container_removed'] = False; code = -2
            elif self.args.backend == 'docker' and code in (0, 2):
                event['reason'] = 'container_identity_missing'; code = -2
            if tag in self.stages:
                stage, final = self.stages.pop(tag)
                try:
                    if code in (0, 2) and (stage / 'result').is_dir():
                        copy_tree_bounded(stage / 'result', final, POLICY['max_report_bytes'], 20000)
                except (OSError, ValueError) as exc:
                    code = -3; event['reason'] = 'unsafe_worker_output:' + type(exc).__name__
                    if final.exists(): shutil.rmtree(final)
                finally:
                    try:
                        shutil.rmtree(stage)
                    except OSError:
                        event['reason'] = 'worker_staging_cleanup_failed'; code = -2
            event['returned_code'] = code
            event['elapsed_seconds'] = round(time.monotonic() - started, 4)
            self.events.append(event)
            dump_json(self.output / 'worker_events.json', self.events)
        return code
