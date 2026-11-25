#!/usr/local/bin/python3
"""Headless CLI adapter for the pinned official bpy wheel (no Blender UI).

Supported: --version, --background, --factory-startup, --disable-autoexec,
--python-exit-code, --python, --python-expr, .blend input, and script args after --.
"""
import runpy
import sys
import traceback
from pathlib import Path


def main():
    import bpy
    arguments = sys.argv[1:]
    if '--version' in arguments:
        print('Blender ' + bpy.app.version_string + ' (official bpy headless runtime)')
        return 0
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    exit_code = 3
    script = None
    expression = None
    blend = None
    index = 0
    while index < len(arguments):
        flag = arguments[index]
        if flag == '--':
            break
        if flag in ('--background', '-b', '--disable-autoexec'):
            pass
        elif flag == '--factory-startup':
            bpy.ops.wm.read_factory_settings(use_empty=False)
        elif flag in ('--python', '--python-expr', '--python-exit-code'):
            index += 1
            if index >= len(arguments):
                raise ValueError('missing CLI argument for ' + flag)
            if flag == '--python':
                script = arguments[index]
            elif flag == '--python-expr':
                expression = arguments[index]
            else:
                exit_code = int(arguments[index])
        elif flag.lower().endswith('.blend'):
            blend = flag
        else:
            raise ValueError('unsupported headless CLI argument: ' + flag)
        index += 1
    try:
        if blend:
            bpy.ops.wm.open_mainfile(filepath=blend, use_scripts=False, load_ui=False)
        if script:
            sys.path.insert(0, str(Path(script).resolve().parent))
            runpy.run_path(script, run_name='__main__')
        if expression:
            exec(compile(expression, '<blender-python-expr>', 'exec'), {'bpy': bpy, '__name__': '__main__'})
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else exit_code)
    except Exception:
        traceback.print_exc()
        return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
