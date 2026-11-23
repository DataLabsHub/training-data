#!/usr/bin/env python3
"""Canonicalize BER CMS objects and reproduce the signing appliance's checks.

Only the input corpus is read.  The acceptance walk deliberately uses original
TLVs, not the DER output: its certificate SET comparison is a raw, strictly
increasing comparison, and its treatment of indefinite lengths is not DER's.
OCTET STRING / BIT STRING values are opaque to DER encoding.  The appliance's
additional inspection of extension octets does not change those abstract values.
"""

import argparse
import base64
import json
from pathlib import Path


def der_length(n):
    if n < 128:
        return bytes((n,))
    b = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    return bytes((128 | len(b),)) + b


def base128(n):
    parts = [n & 127]
    n >>= 7
    while n:
        parts.append(128 | (n & 127))
        n >>= 7
    return bytes(reversed(parts))


def identifier(cls, constructed, tag):
    first = cls | (32 if constructed else 0)
    if tag < 31:
        return bytes((first | tag,))
    return bytes((first | 31,)) + base128(tag)


def minimal_integer(value):
    i = 0
    while i + 1 < len(value):
        if value[i] == 0 and value[i + 1] < 128:
            i += 1
        elif value[i] == 255 and value[i + 1] >= 128:
            i += 1
        else:
            break
    return value[i:]


class TLV:
    __slots__ = ('start', 'content', 'value_end', 'end', 'cls', 'tag',
                 'constructed', 'indefinite', 'minimal_length')

    def is_type(self, tag, cls=0):
        return self.cls == cls and self.tag == tag


class BER:
    """Offset-preserving, lazy BER reader (including nested indefinite TLVs)."""

    def __init__(self, data, appliance=False):
        self.data = data
        self.appliance = appliance
        self.cache = {}

    def header(self, start):
        data = self.data
        if start >= len(data):
            raise ValueError('missing tag')
        n = TLV()
        n.start = start
        first = data[start]
        n.cls = first & 192
        n.constructed = bool(first & 32)
        n.tag = first & 31
        p = start + 1
        if n.tag == 31:
            n.tag = 0
            count = 0
            while True:
                if p >= len(data) or (self.appliance and count == 6):
                    raise ValueError('invalid high tag')
                b = data[p]
                p += 1
                count += 1
                n.tag = (n.tag << 7) | (b & 127)
                if b < 128:
                    break
        if p >= len(data):
            raise ValueError('missing length')
        first_length = data[p]
        p += 1
        n.indefinite = first_length == 128
        n.minimal_length = True
        size = first_length
        if first_length > 128:
            count = first_length & 127
            if p + count > len(data) or (self.appliance and count > 8):
                raise ValueError('invalid length')
            size = int.from_bytes(data[p:p + count], 'big')
            n.minimal_length = size >= 128 and data[p] != 0
            p += count
        n.content = p
        n.value_end = n.end = None if n.indefinite else p + size
        if n.end is not None and n.end > len(data):
            raise ValueError('truncated value')
        return n

    def read(self, start=0):
        if start in self.cache:
            return self.cache[start]
        root = self.header(start)
        if not root.indefinite:
            self.cache[start] = root
            return root
        # Definite children can be skipped; indefinite children need their EOC.
        pending = [root]
        p = root.content
        while pending:
            if p + 2 > len(self.data):
                raise ValueError('missing end of contents')
            if self.data[p:p + 2] == b'\x00\x00':
                n = pending.pop()
                n.value_end = p
                n.end = p + 2
                self.cache[n.start] = n
                p += 2
            else:
                n = self.cache.get(p)
                if n is None:
                    n = self.header(p)
                if n.end is None:
                    pending.append(n)
                    p = n.content
                else:
                    self.cache[p] = n
                    p = n.end
        return root

    def children(self, n, tolerant=False):
        result = []
        p = n.content
        while p < n.value_end:
            try:
                c = self.read(p)
            except ValueError:
                if tolerant:
                    break
                raise
            if c.is_type(0) and not c.constructed and c.content == c.value_end:
                break
            if not tolerant and c.end > n.value_end:
                raise ValueError('child exceeds enclosing value')
            result.append(c)
            p = c.end
        return result

    def value(self, n):
        return self.data[n.content:n.value_end]


def appliance_verdict(data):
    """Mirror the appliance's region discovery and original-byte inspection."""
    reader = BER(data, appliance=True)
    certificates = []
    payloads = []

    def children(n):
        return reader.children(n, tolerant=True)

    def inside(offset, regions):
        return any(start <= offset < end for start, end in regions)

    try:
        root = reader.read()
    except ValueError:
        # A failed TLV read terminates this appliance's walk without a finding.
        return None
    # ContentInfo's second component, then its explicit content's first child.
    ci = children(root) if root.constructed else []
    explicit = children(ci[1]) if len(ci) > 1 and ci[1].constructed else []
    sd = explicit[0] if explicit and explicit[0].constructed else None
    cert_nodes = []
    if sd is not None:
        for c in children(sd):
            if c.is_type(0, 128) and c.constructed:
                for certificate in children(c):
                    if len(certificates) == 16:
                        break
                    certificates.append((certificate.start, certificate.end))
                    cert_nodes.append(certificate)
            if len(certificates) == 16:
                break
    # Extension contents override the otherwise permissive certificate region.
    for certificate in cert_nodes:
        parts = children(certificate) if certificate.constructed else []
        if not parts or not parts[0].constructed:
            continue
        for field in children(parts[0]):
            if not (field.is_type(3, 128) and field.constructed):
                continue
            explicit_exts = children(field)
            if not explicit_exts:
                continue
            for extension in children(explicit_exts[0]):
                if len(payloads) == 128:
                    break
                fields = children(extension)
                if fields and fields[-1].is_type(4):
                    value = fields[-1]
                    payloads.append((value.content, value.value_end))

    first_rejection = None
    pending = [(root, None, 0, False)]
    while pending:
        n, parent_tag, index, open_octets = pending.pop()
        strict = inside(n.start, payloads) or not inside(n.start, certificates)
        bad = False
        if strict:
            bad = not n.minimal_length
            if n.cls == 0:
                if n.constructed and n.tag in (3, 4):
                    bad = True
                elif not n.constructed:
                    value = reader.value(n)
                    if n.tag == 2:
                        bad |= not value or value != minimal_integer(value)
                    elif n.tag == 1 and parent_tag == 16 and index == 0:
                        bad |= value == b'\x00'
        child_nodes = children(n) if n.constructed else []
        if not strict and n.is_type(17) and n.constructed:
            previous = None
            for c in child_nodes:
                raw = data[c.start:c.end]
                if previous is not None and previous >= raw:
                    bad = True
                    break
                previous = raw
        if bad and (first_rejection is None or n.start < first_rejection):
            first_rejection = n.start
        parent = n.tag if n.cls == 0 else None
        pending.extend((c, parent, i, True) for i, c in enumerate(child_nodes))
        # The appliance opens primitive extension octets as TLVs.  It does not
        # reassemble a segmented extnValue before inspecting its segments.
        if (open_octets and n.is_type(4) and not n.constructed and n.content < n.value_end
                and inside(n.content, payloads)
                and inside(n.value_end - 1, payloads)):
            try:
                embedded = reader.read(n.content)
            except ValueError:
                continue
            pending.append((embedded, None, 0, False))
    return first_rejection


class DEREncoder:
    """Schema annotations supplement universal-tag BER canonicalization.

    CMS has implicitly tagged SET OFs, and X.509 has DEFAULT fields which
    cannot be recognized merely from an ASN.1 tag.  These annotations never
    affect the independent acceptance walk above.
    """

    def __init__(self, data):
        self.reader = BER(data)
        self.sets = set()
        self.omit = set()
        self.implicit_strings = {}
        self.nodes = {}
        self.encoded = {}
        self.contents = {}
        self.children = {}

    def integer(self, n):
        return int.from_bytes(self.reader.value(n), 'big', signed=True)

    def annotate_extensions(self, explicit):
        for sequence in self.children[explicit.start][:1]:
            for extension in self.children[sequence.start]:
                fields = self.children[extension.start]
                if (len(fields) >= 3 and fields[0].is_type(6)
                        and fields[1].is_type(1)
                        and self.reader.value(fields[1]) == b'\x00'):
                    self.omit.add(fields[1].start)

    def annotate_certificate(self, cert):
        parts = self.children[cert.start]
        if not parts or not parts[0].is_type(16):
            return
        tbs = parts[0]
        fields = self.children[tbs.start]
        if fields and fields[0].is_type(0, 128):
            version = self.children[fields[0].start]
            if version and version[0].is_type(2) and self.integer(version[0]) == 0:
                self.omit.add(fields[0].start)
        for field in fields:
            if field.cls == 128:
                if field.tag in (1, 2):
                    self.implicit_strings[field.start] = 3
                elif field.tag == 3:
                    self.annotate_extensions(field)

    def annotate_crl(self, crl):
        parts = self.children[crl.start]
        if not parts:
            return
        for field in self.children[parts[0].start]:
            if field.is_type(0, 128):
                self.annotate_extensions(field)
            # RevokedCertificate entries also carry an Extensions sequence.
            if field.is_type(16):
                for entry in self.children[field.start]:
                    fields = self.children[entry.start]
                    if len(fields) > 2 and fields[0].is_type(2):
                        for ext in self.children[fields[2].start]:
                            f = self.children[ext.start]
                            if len(f) > 2 and f[1].is_type(1) and self.reader.value(f[1]) == b'\0':
                                self.omit.add(f[1].start)

    def annotate_cms(self, root):
        fields = self.children[root.start]
        if len(fields) < 2:
            return
        # signedData (1.2.840.113549.1.7.2), including nested ContentInfos.
        if not (fields[0].is_type(6) and self.reader.value(fields[0]) == bytes.fromhex('2a864886f70d010702')):
            return
        wrapper = self.children[fields[1].start]
        if not wrapper:
            return
        sd = self.children[wrapper[0].start]
        for field in sd[3:]:
            if field.is_type(0, 128):
                self.sets.add(field.start)
                for cert in self.children[field.start]:
                    if cert.is_type(16):
                        self.annotate_certificate(cert)
            elif field.is_type(1, 128):
                self.sets.add(field.start)
                for crl in self.children[field.start]:
                    if crl.is_type(16):
                        self.annotate_crl(crl)
            elif field.is_type(17):
                for signer in self.children[field.start]:
                    components = self.children[signer.start]
                    if len(components) > 1 and components[1].is_type(0, 128):
                        self.implicit_strings[components[1].start] = 4
                    for component in components[3:]:
                        if component.cls == 128 and component.tag in (0, 1):
                            self.sets.add(component.start)

    def annotate_algorithms(self, n):
        """DEFAULT parameters in RSA-PSS and RSA-OAEP AlgorithmIdentifiers."""
        fields = self.children[n.start]
        if len(fields) != 2 or not fields[0].is_type(6) or not fields[1].is_type(16):
            return
        oid = self.reader.value(fields[0])
        pss = bytes.fromhex('2a864886f70d01010a')
        oaep = bytes.fromhex('2a864886f70d010107')
        if oid not in (pss, oaep):
            return
        def algorithm(value, wanted_oid, parameter):
            parts = self.children[value.start]
            return (value.is_type(16) and len(parts) == 2
                    and parts[0].is_type(6)
                    and self.reader.value(parts[0]) == wanted_oid
                    and parameter(parts[1]))

        def sha1(value):
            return algorithm(value, bytes.fromhex('2b0e03021a'),
                             lambda p: p.is_type(5))

        def empty_octets(value):
            pending = [value]
            while pending:
                part = pending.pop()
                if not part.is_type(4):
                    return False
                if part.constructed:
                    pending.extend(self.children[part.start])
                elif self.reader.value(part):
                    return False
            return True

        defaults = {
            0: sha1,
            1: lambda v: algorithm(v, bytes.fromhex('2a864886f70d010108'), sha1),
        }
        if oid == pss:
            defaults[2] = lambda v: v.is_type(2) and self.integer(v) == 20
            defaults[3] = lambda v: v.is_type(2) and self.integer(v) == 1
        else:
            defaults[2] = lambda v: algorithm(
                v, bytes.fromhex('2a864886f70d010109'),
                empty_octets)
        for field in self.children[fields[1].start]:
            values = self.children[field.start]
            if field.cls == 128 and len(values) == 1 and field.tag in defaults:
                if defaults[field.tag](values[0]):
                    self.omit.add(field.start)

    def encode(self):
        root = self.reader.read()
        pending = [root]
        order = []
        while pending:
            n = pending.pop()
            self.nodes[n.start] = n
            order.append(n)
            children = self.reader.children(n) if n.constructed else []
            self.children[n.start] = children
            pending.extend(children)
        # All structure is now available for schema-dependent annotations.
        for n in order:
            if n.is_type(16):
                self.annotate_cms(n)
                self.annotate_algorithms(n)
        for n in reversed(order):
            tag = n.tag
            constructed = n.constructed
            string_type = self.implicit_strings.get(n.start)
            if n.cls == 0 and tag in (3, 4, 12, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30):
                string_type = tag
            if string_type is not None:
                if constructed:
                    segments = [self.contents[c.start] for c in self.children[n.start]]
                    if string_type == 3:
                        value = self.join_bits(segments)
                    else:
                        value = b''.join(segments)
                    constructed = False
                else:
                    value = self.reader.value(n)
                if string_type == 3 and value:
                    unused = value[0]
                    if unused and len(value) > 1:
                        value = value[:-1] + bytes((value[-1] & (255 << unused),))
            elif constructed:
                members = [self.encoded[c.start] for c in self.children[n.start]
                           if c.start not in self.omit]
                if n.is_type(17) or n.start in self.sets:
                    members.sort()
                value = b''.join(members)
            else:
                value = self.reader.value(n)
                if n.cls == 0:
                    if tag in (2, 10):
                        value = minimal_integer(value)
                    elif tag == 1:
                        value = b'\xff' if any(value) else b'\x00'
                    elif tag in (6, 13):
                        arcs = []
                        arc = 0
                        for b in value:
                            arc = (arc << 7) | (b & 127)
                            if b < 128:
                                arcs.append(base128(arc))
                                arc = 0
                        value = b''.join(arcs)
            self.contents[n.start] = value
            self.encoded[n.start] = identifier(n.cls, constructed, tag) + der_length(len(value)) + value
        return self.encoded[root.start]

    @staticmethod
    def join_bits(segments):
        # BER permits unused bits only in the final nonempty segment.  This
        # also handles empty strings and arbitrarily nested segmentation.
        parts = [s for s in segments if s]
        if not parts:
            return b'\x00'
        return parts[-1][:1] + b''.join(s[1:] for s in parts)


def adjudicate(data):
    offset = appliance_verdict(data)
    canonical = DEREncoder(data).encode()
    return canonical, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, type=Path)
    args = parser.parse_args()
    for path in sorted(args.corpus.glob('*.der'), key=lambda p: p.name):
        if not path.is_file():
            continue
        canonical, offset = adjudicate(path.read_bytes())
        print(json.dumps({
            'object': path.name,
            'canonical_der': base64.b64encode(canonical).decode('ascii'),
            'verdict': 'accept' if offset is None else 'reject',
            'first_rejecting_element': offset,
        }, separators=(',', ':')))


if __name__ == '__main__':
    main()
