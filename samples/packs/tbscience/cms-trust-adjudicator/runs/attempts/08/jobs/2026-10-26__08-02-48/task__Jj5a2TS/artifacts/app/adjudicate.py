#!/usr/bin/env python3
"""Canonicalise CMS SignedData and reproduce the signing appliance's TLV checks.

The two operations deliberately use different views of the input.  DER encoding
uses the CMS/X.509 schema, whereas the appliance checks original TLVs, including
original byte ordering and original offsets.  Neither operation verifies a
cryptographic signature or consults a trust store.
"""
import argparse
import base64
import json
from pathlib import Path
import sys
from dataclasses import dataclass, field


class EncodingError(ValueError):
    pass


def der_length(n):
    if n < 128:
        return bytes((n,))
    b = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    return bytes((128 | len(b),)) + b


def base128(n):
    b = [n & 127]
    n >>= 7
    while n:
        b.append(128 | (n & 127))
        n >>= 7
    return bytes(reversed(b))


def der_tag(cls, constructed, tag):
    first = cls | (32 if constructed else 0)
    return (bytes((first | tag,)) if tag < 31
            else bytes((first | 31,)) + base128(tag))


def integer_contents(value):
    i = 0
    while i + 1 < len(value):
        if ((value[i] == 0 and value[i + 1] < 128) or
                (value[i] == 255 and value[i + 1] >= 128)):
            i += 1
        else:
            break
    return value[i:]


@dataclass
class TLV:
    start: int
    head: int
    stop: int
    end: int
    cls: int
    constructed: bool
    tag: int
    indefinite: bool
    minimal_length: bool

    @property
    def length(self):
        return self.stop - self.head

    def is_tag(self, tag, cls=0):
        return self.cls == cls and self.tag == tag

    def is_eoc(self):
        return self.is_tag(0) and not self.constructed and self.length == 0


class Reader:
    """Keep source offsets even when a definite TLV has nonminimal lengths."""
    def __init__(self, data, appliance=False):
        self.data = data
        self.appliance = appliance
        self.cache = {}

    def read(self, start):
        if start in self.cache:
            return self.cache[start]
        d, size, p = self.data, len(self.data), start
        if p >= size:
            raise EncodingError('missing tag')
        first = d[p]
        p += 1
        cls, constructed, tag = first & 192, bool(first & 32), first & 31
        if tag == 31:
            tag, count = 0, 0
            while True:
                if p >= size or (self.appliance and count == 6):
                    raise EncodingError('invalid high tag')
                b = d[p]
                p += 1
                count += 1
                tag = (tag << 7) | (b & 127)
                if not b & 128:
                    break
        if p >= size:
            raise EncodingError('missing length')
        lb = d[p]
        p += 1
        indefinite, minimal = lb == 128, True
        if indefinite:
            head = p
            while p + 2 <= size and d[p:p + 2] != b'\x00\x00':
                p = self.read(p).end
            if p + 2 > size:
                raise EncodingError('missing end of contents')
            stop, end = p, p + 2
        else:
            if lb & 128:
                count = lb & 127
                if (p + count > size or
                        (self.appliance and count > 8)):
                    raise EncodingError('invalid length')
                length = int.from_bytes(d[p:p + count], 'big')
                minimal = length >= 128 and d[p] != 0
                p += count
            else:
                length = lb
            head, stop, end = p, p + length, p + length
            if end > size:
                raise EncodingError('truncated value')
        n = TLV(start, head, stop, end, cls, constructed, tag,
                indefinite, minimal)
        self.cache[start] = n
        return n

    def children(self, n):
        p = n.head
        while p < n.stop:
            child = self.read(p)
            if child.is_eoc():
                return
            yield child
            p = child.end

    def contents(self, n):
        return self.data[n.head:n.stop]


class Appliance:
    """The appliance's three structural regions, without DER assumptions.

    * CMS: minimal definite lengths, primitive strings, minimal INTEGERs;
      indefinite lengths are accepted. A first FALSE in a SEQUENCE is rejected.
    * Certificate TLVs: only universal SET ordering is checked, using the
      original complete encodings, in strictly increasing order.
    * extnValue contents: the CMS checks apply again. The walker
      follows nested OCTET STRING contents only within these recorded ranges.

    Region discovery has the same 16-certificate/128-extension limits as the
    appliance. It does not use certificate identities or extension OIDs.
    """
    def __init__(self, data):
        self.r = Reader(data, appliance=True)
        self.certificates = []
        self.extensions = []
        self.first = None

    @staticmethod
    def inside(offset, ranges):
        return any(start <= offset < end for start, end in ranges)

    def discover(self):
        r = self.r
        try:
            root = r.read(0)
            if not root.constructed:
                return
            wrapper = list(r.children(root))[1]
            if not wrapper.constructed:
                return
            signed = next(r.children(wrapper))
            if not signed.constructed:
                return
            for component in r.children(signed):
                if component.is_tag(0, 128) and component.constructed:
                    for cert in r.children(component):
                        if len(self.certificates) == 16:
                            break
                        self.certificates.append((cert.start, cert.end))
                if len(self.certificates) == 16:
                    break
        except (EncodingError, IndexError, StopIteration):
            pass
        for start, _ in self.certificates:
            try:
                cert = r.read(start)
                if not cert.constructed:
                    continue
                tbs = next(r.children(cert))
                if not tbs.constructed:
                    continue
                for component in r.children(tbs):
                    if not (component.is_tag(3, 128) and component.constructed):
                        continue
                    extensions = next(r.children(component))
                    for extension in r.children(extensions):
                        if len(self.extensions) == 128:
                            break
                        fields = list(r.children(extension))
                        if fields:
                            value = fields[-1]
                            if value.is_tag(4):
                                self.extensions.append((value.head, value.stop))
            except (EncodingError, StopIteration):
                continue

    def reject(self, offset):
        if self.first is None or offset < self.first:
            self.first = offset

    def check(self, n, parent_tag=None, index=0):
        r = self.r
        strict = (self.inside(n.start, self.extensions) or
                  not self.inside(n.start, self.certificates))
        if not strict:
            if n.is_tag(17) and n.constructed:
                previous = None
                try:
                    for child in r.children(n):
                        raw = r.data[child.start:child.end]
                        if previous is not None and previous >= raw:
                            self.reject(n.start)
                            break
                        previous = raw
                except EncodingError:
                    pass
            return
        if not n.minimal_length:
            self.reject(n.start)
        if n.cls == 0 and n.constructed and n.tag in (3, 4):
            self.reject(n.start)
        if n.cls == 0 and not n.constructed:
            value = r.contents(n)
            if n.tag == 2 and (not value or integer_contents(value) != value):
                self.reject(n.start)
            if (n.tag == 1 and parent_tag == 16 and index == 0 and
                    value == b'\x00'):
                self.reject(n.start)

    def walk_root(self, start):
        try:
            root = self.r.read(start)
        except EncodingError:
            return
        self.check(root)
        if root.constructed:
            self.walk_children(root)

    def walk_children(self, parent):
        r = self.r
        try:
            for i, n in enumerate(r.children(parent)):
                self.check(n, parent.tag if parent.cls == 0 else None, i)
                if n.constructed:
                    self.walk_children(n)
                if (n.is_tag(4) and not n.constructed and n.length and
                        self.inside(n.head, self.extensions) and
                        self.inside(n.stop - 1, self.extensions)):
                    self.walk_root(n.head)
        except EncodingError:
            return

    def verdict(self):
        self.discover()
        self.walk_root(0)
        return self.first


@dataclass
class Value:
    cls: int
    constructed: bool
    tag: int
    contents: bytes = b''
    children: list = field(default_factory=list)
    set_of: bool = False
    omit: bool = False
    string_type: int = 0

    def is_tag(self, tag, cls=0):
        return self.cls == cls and self.tag == tag


def abstract_tree(reader, n):
    v = Value(n.cls, n.constructed, n.tag, reader.contents(n))
    if n.constructed:
        v.children = [abstract_tree(reader, c) for c in reader.children(n)]
    v.set_of = n.is_tag(17)
    return v


def integer(v):
    return int.from_bytes(v.contents, 'big', signed=True)


def oid(v):
    if not v.is_tag(6):
        return ()
    arcs, n = [], 0
    for b in v.contents:
        n = (n << 7) | (b & 127)
        if not b & 128:
            arcs.append(n)
            n = 0
    if not arcs:
        return ()
    a = min(arcs[0] // 40, 2)
    return (a, arcs[0] - 40 * a, *arcs[1:])


def string_contents(v, kind):
    """Flatten nested BER string fragments without interpreting their bytes."""
    if not v.constructed:
        return v.contents
    parts = [string_contents(c, kind) for c in v.children]
    if kind != 3:
        return b''.join(parts)
    if not parts:
        return b'\x00'
    # BER requires every nonfinal fragment to have zero unused bits.
    if any(not p or p[0] != 0 for p in parts[:-1]) or not parts[-1]:
        raise EncodingError('invalid BIT STRING fragmentation')
    return parts[-1][:1] + b''.join(p[1:] for p in parts)


# DER forbids constructed encodings for these universal string types.
STRING_TYPES = {3, 4, 12, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30}


def encode(v):
    if v.omit:
        return b''
    constructed = v.constructed
    kind = v.tag if v.cls == 0 else v.string_type
    if kind in STRING_TYPES:
        content = string_contents(v, kind)
        constructed = False
        if kind == 3 and content and content[0] and len(content) > 1:
            content = content[:-1] + bytes((content[-1] & (255 << content[0]),))
    elif constructed:
        parts = [encode(c) for c in v.children if not c.omit]
        if v.set_of:
            parts.sort()
        content = b''.join(parts)
    else:
        content = v.contents
        if v.cls == 0:
            if v.tag in (2, 10):
                content = integer_contents(content)
            elif v.tag == 1:
                content = b'\xff' if any(content) else b'\x00'
            elif v.tag in (6, 13):
                arcs, n = [], 0
                for b in content:
                    n = (n << 7) | (b & 127)
                    if not b & 128:
                        arcs.append(base128(n))
                        n = 0
                content = b''.join(arcs)
    return der_tag(v.cls, constructed, v.tag) + der_length(len(content)) + content


SIGNED_DATA = (1, 2, 840, 113549, 1, 7, 2)
COUNTER_SIGNATURE = (1, 2, 840, 113549, 1, 9, 6)
RSA_PSS = (1, 2, 840, 113549, 1, 1, 10)
RSA_OAEP = (1, 2, 840, 113549, 1, 1, 7)
MGF1 = (1, 2, 840, 113549, 1, 1, 8)
SHA1 = bytes.fromhex('300906052b0e03021a0500')
MGF1_SHA1 = bytes.fromhex('301606092a864886f70d010108') + SHA1
P_SPECIFIED_EMPTY = bytes.fromhex('300d06092a864886f70d0101090400')


def algorithm(v):
    if not v.is_tag(16) or not v.children:
        return
    name = oid(v.children[0])
    if len(v.children) < 2:
        return
    params = v.children[1]
    if name == MGF1:
        algorithm(params)
    elif name in (RSA_PSS, RSA_OAEP) and params.is_tag(16):
        defaults = ({0: SHA1, 1: MGF1_SHA1, 2: b'\x02\x01\x14',
                     3: b'\x02\x01\x01'} if name == RSA_PSS else
                    {0: SHA1, 1: MGF1_SHA1, 2: P_SPECIFIED_EMPTY})
        for field_value in params.children:
            if field_value.cls == 128 and len(field_value.children) == 1:
                inner = field_value.children[0]
                algorithm(inner)
                if encode(inner) == defaults.get(field_value.tag):
                    field_value.omit = True


def extensions(v):
    for ext in v.children:
        c = ext.children
        if len(c) >= 3 and c[1].is_tag(1) and not any(c[1].contents):
            c[1].omit = True
        # extnValue is an OCTET STRING, not an ASN.1 open type. Its octets
        # (including any embedded ASN.1 encoding) are part of its abstract
        # value and must be preserved. Constructed fragments are flattened.


def certificate(v):
    if not v.is_tag(16) or not v.children:
        return
    tbs = v.children[0]
    if not tbs.is_tag(16):
        return
    c = tbs.children
    shift = 0
    if c and c[0].is_tag(0, 128):
        shift = 1
        if (len(c[0].children) == 1 and c[0].children[0].is_tag(2)
                and integer(c[0].children[0]) == 0):
            c[0].omit = True
    if len(c) > shift + 1:
        algorithm(c[shift + 1])
    if len(c) > shift + 5 and c[shift + 5].children:
        algorithm(c[shift + 5].children[0])
    for component in c[shift + 6:]:
        if component.cls == 128 and component.tag in (1, 2):
            component.string_type = 3  # IMPLICIT BIT STRING unique IDs
        if component.is_tag(3, 128) and component.children:
            extensions(component.children[0])
    if len(v.children) > 1:
        algorithm(v.children[1])


def crl(v):
    if not v.is_tag(16) or not v.children:
        return
    c = v.children[0].children
    shift = int(bool(c and c[0].is_tag(2)))
    if len(c) > shift:
        algorithm(c[shift])
    for component in c:
        if component.is_tag(0, 128) and component.children:
            extensions(component.children[0])
    # revokedCertificates entries have an optional Extensions field.
    for component in c[shift + 3:]:
        if component.is_tag(16):
            for entry in component.children:
                if len(entry.children) >= 3:
                    extensions(entry.children[2])
    if len(v.children) > 1:
        algorithm(v.children[1])


def attributes(v):
    v.set_of = True
    for attribute in v.children:
        c = attribute.children
        if len(c) < 2:
            continue
        c[1].set_of = True
        if oid(c[0]) == COUNTER_SIGNATURE:
            for signer in c[1].children:
                signer_info(signer)


def signer_info(v):
    c = v.children
    if len(c) > 1 and c[1].is_tag(0, 128):
        c[1].string_type = 4  # SubjectKeyIdentifier, IMPLICIT OCTET STRING
    if len(c) > 2:
        algorithm(c[2])
    i = 3
    if len(c) > i and c[i].is_tag(0, 128):
        attributes(c[i])
        i += 1
    if len(c) > i:
        algorithm(c[i])
    for component in c[i + 2:]:
        if component.is_tag(1, 128):
            attributes(component)


def cms_schema(root):
    if len(root.children) < 2 or oid(root.children[0]) != SIGNED_DATA:
        return
    wrapper = root.children[1]
    if not wrapper.children:
        return
    signed = wrapper.children[0]
    c = signed.children
    if len(c) > 1:
        c[1].set_of = True
        for alg in c[1].children:
            algorithm(alg)
    for component in c[3:]:
        if component.is_tag(0, 128):
            component.set_of = True
            for cert in component.children:
                certificate(cert)
        elif component.is_tag(1, 128):
            component.set_of = True
            for entry in component.children:
                crl(entry)
        elif component.is_tag(17):
            component.set_of = True
            for signer in component.children:
                signer_info(signer)


def adjudicate(data):
    rejecting = Appliance(data).verdict()
    reader = Reader(data)
    tree = abstract_tree(reader, reader.read(0))
    cms_schema(tree)
    return encode(tree), rejecting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, type=Path)
    args = parser.parse_args()
    sys.setrecursionlimit(100000)
    for path in sorted(args.corpus.glob('*.der'), key=lambda p: p.name):
        if not path.is_file():
            continue
        canonical, offset = adjudicate(path.read_bytes())
        record = {'object': path.name,
                  'canonical_der': base64.b64encode(canonical).decode('ascii'),
                  'verdict': 'accept' if offset is None else 'reject',
                  'first_rejecting_element': offset}
        print(json.dumps(record, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
