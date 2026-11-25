"""Validation and geometry-only camera planning; no Blender dependency."""
import math
import re


def vector(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(label + ' must contain three finite numbers')
    result = [float(v) for v in value]
    if not all(math.isfinite(v) for v in result):
        raise ValueError(label + ' must contain three finite numbers')
    return result


def validate(request):
    if request.get('action') != 'render.views':
        raise ValueError('action must be render.views')
    args = dict(request.get('args') or {})
    resolution = args.get('resolution', [1448, 1086])
    if not isinstance(resolution, list) or len(resolution) != 2 or any(type(v) is not int or not 16 <= v <= 8192 for v in resolution):
        raise ValueError('resolution must be two integers in [16,8192]')
    args['resolution'] = resolution
    samples = args.get('samples', 64)
    if type(samples) is not int or not 1 <= samples <= 4096:
        raise ValueError('samples must be in [1,4096]')
    args['samples'] = samples
    if args.get('engine', 'CYCLES') not in ('CYCLES', 'BLENDER_EEVEE_NEXT'):
        raise ValueError('unsupported engine')
    args.setdefault('engine', 'CYCLES')
    if type(args.get('gray_override', False)) is not bool:
        raise ValueError('gray_override must be boolean')
    mode = args.setdefault('mode', 'explicit' if args.get('views') else 'auto')
    if mode not in ('auto', 'explicit'):
        raise ValueError('mode must be auto or explicit')
    names = set()
    entries = args.get('views', []) if mode == 'explicit' else args.get('groups', [])
    if not isinstance(entries, list) or len(entries) > 32 or (mode == 'explicit' and not entries):
        raise ValueError('provide 1..32 explicit views or up to 32 auto groups')
    for item in entries:
        name = item.get('name', '')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name) or name in names or (mode == 'auto' and name in ('overview', 'reverse')):
            raise ValueError('view names must be unique safe filenames; auto reserves overview/reverse')
        names.add(name)
        if mode == 'explicit':
            item['location'] = vector(item.get('location'), 'location')
            item['target'] = vector(item.get('target'), 'target')
            if math.dist(item['location'], item['target']) < 1e-9:
                raise ValueError('camera location and target must differ')
        else:
            patterns = item.get('objects')
            if not isinstance(patterns, list) or not patterns or not all(isinstance(p, str) and p for p in patterns):
                raise ValueError('auto group objects must be a nonempty list of glob patterns')
        lens = float(item.get('lens', 50))
        if not math.isfinite(lens) or not 1 <= lens <= 500:
            raise ValueError('lens must be in [1,500]')
        item['lens'] = lens
    limit = args.setdefault('max_closeups', 4)
    if type(limit) is not int or not 0 <= limit <= 16:
        raise ValueError('max_closeups must be in [0,16]')
    exclusions = args.setdefault('exclude_objects', ['Ground', 'Terrain', 'Landscape', 'Water', 'Lake', 'Sky*'])
    if not isinstance(exclusions, list) or not all(isinstance(p, str) for p in exclusions):
        raise ValueError('exclude_objects must be a list of glob patterns')
    return args


def bounds(points):
    if not points:
        raise ValueError('no renderable geometry for camera planning')
    points = [vector(p, 'bound point') for p in points]
    lo = [min(p[i] for p in points) for i in range(3)]
    hi = [max(p[i] for p in points) for i in range(3)]
    center = [(a + b) / 2 for a, b in zip(lo, hi)]
    radius = max(math.dist(center, p) for p in points)
    return center, max(radius, 0.001)


def fit_camera(points, azimuth, elevation, lens, resolution):
    """Fit a bounding sphere inside both FOV axes with a 15% margin."""
    center, radius = bounds(points)
    aspect = resolution[0] / resolution[1]
    half_fov = math.atan(min(36.0, 36.0 / aspect) / (2 * lens))
    distance = radius / math.sin(half_fov) * 1.15
    a, e = math.radians(azimuth), math.radians(elevation)
    direction = [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
    return {'location': [center[i] + distance * direction[i] for i in range(3)],
            'target': center, 'lens': lens, 'clip_start': max(0.00001, min(0.01, radius / 100)),
            'clip_end': max(100, distance + radius * 4), 'fit_radius': radius}


def visibility_targets(names, boxes, limit=64):
    """Bound the advisory ray heuristic; camera framing still fits every target.

    Retain large components and spread remaining probes across world-space
    centers. This sampling is not a visibility certificate or a quality score.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError('visibility target limit must be a positive integer')
    if len(names) <= limit:
        return list(names)
    metrics={name:bounds(boxes[name]) for name in names}
    largest=sorted(names,key=lambda n:(-metrics[n][1],n))[:min(8,limit)]
    selected=set(largest)
    remaining=sorted((n for n in names if n not in selected),key=lambda n:(tuple(metrics[n][0]),n))
    slots=limit-len(largest)
    if slots==1:largest.append(remaining[len(remaining)//2])
    elif slots>1:largest.extend(remaining[i*(len(remaining)-1)//(slots-1)] for i in range(slots))
    return largest
