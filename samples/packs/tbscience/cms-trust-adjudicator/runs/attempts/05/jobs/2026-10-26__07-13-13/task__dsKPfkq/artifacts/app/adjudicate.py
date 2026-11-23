#!/usr/bin/env python3
"""Canonicalize BER CMS SignedData and reproduce the appliance's TLV policy.

The two operations deliberately have separate implementations.  DER ordering
uses canonical encodings, whereas the certificate reader compares input TLVs.
The CMS reader also inspects extension payloads, without making those payload
bytes part of the enclosing OCTET STRING's DER transformation.

Only the files selected by --corpus are read.  No trust database, external
executable, or stored verdicts are needed.
"""

import argparse
import base64
import datetime
import json
from pathlib import Path
import re
import sys


class EncodingError(ValueError):
    pass


def length_bytes(n):
    if n < 128:
        return bytes((n,))
    value = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    return bytes((128 + len(value),)) + value


def base128(n):
    parts = [n & 127]
    n >>= 7
    while n:
        parts.append(128 | (n & 127))
        n >>= 7
    return bytes(reversed(parts))


def tag_bytes(cls, constructed, number):
    first = cls | (32 if constructed else 0)
    if number < 31:
        return bytes((first | number,))
    return bytes((first | 31,)) + base128(number)


def integer_bytes(value):
    while len(value) > 1 and (
        (value[0] == 0 and value[1] < 128)
        or (value[0] == 255 and value[1] >= 128)
    ):
        value = value[1:]
    return value or b'\x00'


class Element:
    __slots__ = ('start', 'content', 'stop', 'end', 'cls', 'number',
                 'constructed', 'indefinite', 'minimal_length')

    def is_tag(self, number, cls=0):
        return self.cls == cls and self.number == number


class Reader:
    """A lossless TLV reader; offsets always refer to the original input.

    Reading a TLV and traversing its children are separate operations.  This
    matters for the appliance's extra reads inside primitive extension OCTET
    STRINGs: it attempts a TLV read there even when the octets are not ASN.1.
    """

    def __init__(self, data, appliance=False):
        self.data = data
        self.cache = {}
        self.appliance = appliance

    def read(self, start):
        if start in self.cache:
            return self.cache[start]
        data = self.data
        size = len(data)
        if start >= size:
            raise EncodingError('missing tag')
        node = Element()
        node.start = start
        first = data[start]
        node.cls = first & 192
        node.constructed = bool(first & 32)
        node.number = first & 31
        p = start + 1
        if node.number == 31:
            node.number = 0
            tag_start = p
            while True:
                if self.appliance and p - tag_start == 6:
                    raise EncodingError('appliance tag limit')
                if p >= size:
                    raise EncodingError('truncated tag')
                byte = data[p]
                p += 1
                node.number = (node.number << 7) | (byte & 127)
                if byte < 128:
                    break
        if p >= size:
            raise EncodingError('missing length')
        first_length = data[p]
        p += 1
        node.indefinite = first_length == 128
        node.minimal_length = True
        if first_length > 128:
            count = first_length & 127
            if self.appliance and count > 8:
                raise EncodingError('appliance length limit')
            if p + count > size:
                raise EncodingError('truncated length')
            size_bytes = data[p:p + count]
            length = int.from_bytes(size_bytes, 'big')
            node.minimal_length = length >= 128 and size_bytes[0] != 0
            p += count
        else:
            length = first_length
        node.content = p
        if node.indefinite:
            while True:
                if p + 2 > size:
                    raise EncodingError('missing end-of-contents')
                if data[p:p + 2] == b'\x00\x00':
                    break
                p = self.read(p).end
            node.stop = p
            node.end = p + 2
        else:
            node.stop = p + length
            node.end = node.stop
            if node.end > size:
                raise EncodingError('content exceeds input')
        self.cache[start] = node
        return node

    def children(self, node):
        p = node.content
        while p < node.stop:
            child = self.read(p)
            if (child.is_tag(0) and not child.constructed
                    and child.content == child.stop):
                break
            yield child
            p = child.end

    def value(self, node):
        return self.data[node.content:node.stop]

    def raw(self, node):
        return self.data[node.start:node.end]


def appliance_rejection(reader):
    """Reproduce the three reader domains, including their offset precedence.

    The appliance records at most 16 certificate spans and 128 extension-value
    spans.  Extension-value spans override the surrounding certificate domain.
    All spans are discovered from the input layout, before any DER reordering.
    """
    certificates = []
    extensions = []

    def children(node):
        try:
            yield from reader.children(node)
        except EncodingError:
            return

    def child(node, index):
        for i, item in enumerate(children(node)):
            if i == index:
                return item
        return None

    try:
        root = reader.read(0)
    except EncodingError:
        return None
    if root.constructed:
        wrapper = child(root, 1)
        if wrapper is not None and wrapper.constructed:
            signed_data = child(wrapper, 0)
            if signed_data is not None and signed_data.constructed:
                for field in children(signed_data):
                    if field.is_tag(0, 128) and field.constructed:
                        for cert in children(field):
                            if len(certificates) == 16:
                                break
                            certificates.append(cert)
                    if len(certificates) == 16:
                        break
    for cert in certificates:
        if not cert.constructed:
            continue
        tbs = child(cert, 0)
        if tbs is None or not tbs.constructed:
            continue
        for field in children(tbs):
            if not (field.is_tag(3, 128) and field.constructed):
                continue
            sequence = child(field, 0)
            if sequence is None:
                continue
            for extension in children(sequence):
                if len(extensions) == 128:
                    break
                last = None
                for part in children(extension):
                    last = part
                if last is not None and last.is_tag(4):
                    extensions.append((last.content, last.stop))

    def in_extension(offset):
        return any(start <= offset < end for start, end in extensions)

    def in_certificate(offset):
        return any(cert.start <= offset < cert.end for cert in certificates)

    first_bad = None
    pending = [(root, None, 0, False)]
    # The same embedded value can be reached through more than one OCTET layer.
    seen = set()
    while pending:
        node, parent, index, inspect_octets = pending.pop()
        state = (node.start, parent, index, inspect_octets)
        if state in seen:
            continue
        seen.add(state)
        cert_mode = in_certificate(node.start) and not in_extension(node.start)
        bad = False
        if cert_mode:
            if node.is_tag(17) and node.constructed:
                previous = None
                for member in children(node):
                    current = reader.raw(member)
                    if previous is not None and previous >= current:
                        bad = True
                        break
                    previous = current
        else:
            bad = not node.minimal_length
            if node.cls == 0:
                if node.constructed and node.number in (3, 4):
                    bad = True
                elif not node.constructed:
                    value = reader.value(node)
                    if node.number == 2:
                        bad |= not value or integer_bytes(value) != value
                    elif (node.number == 1 and parent == 16 and index == 0
                          and value == b'\x00'):
                        bad = True
        if bad and (first_bad is None or node.start < first_bad):
            first_bad = node.start
        if node.constructed:
            parent_number = node.number if node.cls == 0 else None
            pending.extend((item, parent_number, i, True)
                           for i, item in enumerate(children(node)))
        elif (inspect_octets and node.is_tag(4) and node.content < node.stop
              and in_extension(node.content) and in_extension(node.stop - 1)):
            try:
                pending.append((reader.read(node.content), None, 0, False))
            except EncodingError:
                pass
    return first_bad


# OIDs are represented by their BER content octets; none is a corpus identity.
SIGNED_DATA = bytes.fromhex('2a864886f70d010702')
COUNTERSIGNATURE = bytes.fromhex('2a864886f70d010906')
RSA_PSS = bytes.fromhex('2a864886f70d01010a')
RSA_OAEP = bytes.fromhex('2a864886f70d010107')
SHA1_ALGORITHM = bytes.fromhex('300906052b0e03021a0500')
MGF1_SHA1 = bytes.fromhex('301606092a864886f70d010108300906052b0e03021a0500')
PSPECIFIED_EMPTY = bytes.fromhex('300d06092a864886f70d0101090400')


class Canonicalizer:
    def __init__(self, reader):
        self.reader = reader
        self.sets = set()
        self.omit = set()
        self.implicit_strings = {}
        self.members = {}

    def children(self, node):
        if node.start not in self.members:
            self.members[node.start] = list(self.reader.children(node))
        return self.members[node.start]

    def integer(self, node):
        return int.from_bytes(self.reader.value(node), 'big', signed=True)

    def mark_extensions(self, sequence):
        for extension in self.children(sequence):
            parts = self.children(extension)
            # Extension.critical BOOLEAN DEFAULT FALSE is the second field.
            if (len(parts) >= 3 and parts[0].is_tag(6)
                    and parts[1].is_tag(1) and self.reader.value(parts[1]) == b'\x00'):
                self.omit.add(parts[1].start)
            # extnValue is an OCTET STRING.  Its encoded ASN.1 payload is
            # inspected by the appliance but remains the *value* of this
            # OCTET STRING when canonicalizing the enclosing CMS object.

    def mark_certificate(self, certificate):
        parts = self.children(certificate)
        if not parts:
            return
        fields = self.children(parts[0])
        if fields and fields[0].is_tag(0, 128):
            version = self.children(fields[0])
            if version and version[0].is_tag(2) and self.integer(version[0]) == 0:
                self.omit.add(fields[0].start)
        for field in fields:
            if field.cls == 128:
                if field.number in (1, 2):
                    self.implicit_strings[field.start] = 3  # UniqueIdentifier
                elif field.number == 3 and field.constructed:
                    explicit = self.children(field)
                    if explicit:
                        self.mark_extensions(explicit[0])

    def mark_crl(self, crl):
        parts = self.children(crl)
        if not parts:
            return
        for field in self.children(parts[0]):
            if field.is_tag(0, 128) and field.constructed:
                explicit = self.children(field)
                if explicit:
                    self.mark_extensions(explicit[0])
            elif field.is_tag(16) and field.constructed:
                # A revokedCertificates member has serial, time, extensions.
                for entry in self.children(field):
                    if entry.is_tag(16) and entry.constructed:
                        items = self.children(entry)
                        if len(items) == 3 and items[0].is_tag(2):
                            self.mark_extensions(items[2])

    def mark_attributes(self, field):
        self.sets.add(field.start)
        for attribute in self.children(field):
            parts = self.children(attribute)
            if (len(parts) == 2 and parts[0].is_tag(6)
                    and self.reader.value(parts[0]) == COUNTERSIGNATURE):
                for signer in self.children(parts[1]):
                    self.mark_signer(signer)

    def mark_signer(self, signer):
        fields = self.children(signer)
        if len(fields) > 1 and fields[1].is_tag(0, 128):
            self.implicit_strings[fields[1].start] = 4  # SubjectKeyIdentifier
        for field in fields[2:]:
            if field.cls == 128 and field.number in (0, 1) and field.constructed:
                self.mark_attributes(field)

    def mark_schema(self, root):
        parts = self.children(root)
        if not (len(parts) >= 2 and parts[0].is_tag(6)
                and self.reader.value(parts[0]) == SIGNED_DATA):
            return
        explicit = self.children(parts[1])
        if not explicit:
            return
        fields = self.children(explicit[0])
        for field in fields[3:]:
            if field.is_tag(0, 128):
                self.sets.add(field.start)  # CertificateSet, IMPLICIT SET OF
                for certificate in self.children(field):
                    if certificate.is_tag(16):
                        self.mark_certificate(certificate)
            elif field.is_tag(1, 128):
                self.sets.add(field.start)  # RevocationInfoChoices
                for crl in self.children(field):
                    if crl.is_tag(16):
                        self.mark_crl(crl)
            elif field.is_tag(17):
                for signer in self.children(field):
                    self.mark_signer(signer)

    def string_value(self, node, kind):
        if not node.constructed:
            return self.reader.value(node)
        values = [self.string_value(part, kind) for part in self.children(node)]
        if kind != 3:
            return b''.join(values)
        # Each BIT STRING fragment has its own unused-bit count octet.  Only
        # the final fragment may have unused bits in a valid BER value.
        nonempty = [value for value in values if value]
        if not nonempty:
            return b'\x00'
        return nonempty[-1][:1] + b''.join(value[1:] for value in nonempty)

    def encode(self, root):
        self.mark_schema(root)
        encoded = {}
        stack = [(root, False)]
        string_tags = {3, 4, 12, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30}
        while stack:
            node, ready = stack.pop()
            if node.start in self.omit:
                encoded[node.start] = b''
                continue
            kind = self.implicit_strings.get(node.start)
            if kind is None and node.cls == 0 and node.number in string_tags:
                kind = node.number
            constructed = node.constructed
            if kind is not None:
                value = self.string_value(node, kind)
                constructed = False
                if kind == 3:
                    if not value:
                        value = b'\x00'
                    elif value[0] and len(value) > 1:
                        value = value[:-1] + bytes((value[-1] & (255 << value[0] & 255),))
            elif constructed:
                members = self.children(node)
                if not ready:
                    stack.append((node, True))
                    stack.extend((member, False) for member in reversed(members))
                    continue
                chunks = [encoded[member.start] for member in members]
                # AlgorithmIdentifier parameters are open types with specified
                # defaults for the two RSA parameter SEQUENCEs below.
                if (node.is_tag(16) and len(members) == 2
                        and members[0].is_tag(6) and members[1].is_tag(16)):
                    oid = self.reader.value(members[0])
                    if oid in (RSA_PSS, RSA_OAEP):
                        defaults = {0: SHA1_ALGORITHM, 1: MGF1_SHA1}
                        defaults.update({2: b'\x02\x01\x14', 3: b'\x02\x01\x01'}
                                        if oid == RSA_PSS else {2: PSPECIFIED_EMPTY})
                        params = []
                        for param in self.children(members[1]):
                            inners = self.children(param) if param.constructed else []
                            if (param.cls == 128 and len(inners) == 1
                                    and encoded[inners[0].start] == defaults.get(param.number)):
                                continue
                            params.append(encoded[param.start])
                        body = b''.join(params)
                        chunks[1] = b'\x30' + length_bytes(len(body)) + body
                if node.is_tag(17) or node.start in self.sets:
                    chunks.sort()
                value = b''.join(chunks)
            else:
                value = self.reader.value(node)
                if node.cls == 0:
                    if node.number in (2, 10):
                        value = integer_bytes(value)
                    elif node.number == 1:
                        value = b'\xff' if any(value) else b'\x00'
                    elif node.number in (6, 13):
                        value = canonical_oid(value)
                    elif node.number in (23, 24):
                        value = canonical_time(value, node.number)
            encoded[node.start] = (tag_bytes(node.cls, constructed, node.number)
                                   + length_bytes(len(value)) + value)
        return encoded[root.start]


def canonical_oid(value):
    result = bytearray()
    number = 0
    for byte in value:
        number = (number << 7) | (byte & 127)
        if byte < 128:
            result.extend(base128(number))
            number = 0
    if value and value[-1] >= 128:
        raise EncodingError('unterminated OID component')
    return bytes(result)


def canonical_time(value, kind):
    text = value.decode('ascii')
    pattern = (r'(\d{10})(\d{2})?(Z|[+-]\d{4})' if kind == 23 else
               r'(\d{10})(\d{2})?(\d{2})?([.,]\d+)?(Z|[+-]\d{4})')
    match = re.fullmatch(pattern, text)
    if match is None:
        return value
    fraction = ''
    if kind == 23:
        digits, seconds, zone = match.groups()
        year = int(digits[:2])
        year += 1900 if year >= 50 else 2000
        month, day, hour, minute = [int(digits[i:i + 2]) for i in (2, 4, 6, 8)]
        second = int(seconds or '0')
    else:
        digits, minutes, seconds, fractional, zone = match.groups()
        year = int(digits[:4])
        month, day, hour = [int(digits[i:i + 2]) for i in (4, 6, 8)]
        minute, second = int(minutes or '0'), int(seconds or '0')
        if fractional:
            scale = 1 if seconds else (60 if minutes else 3600)
            numerator = int(fractional[1:]) * scale
            denominator = 10 ** (len(fractional) - 1)
            whole, remainder = divmod(numerator, denominator)
            second += whole
            fraction = str(remainder).zfill(len(fractional) - 1).rstrip('0')
    # Keep a canonical leap second intact; datetime does not represent it.
    if second == 60 and zone == 'Z' and (kind == 23 or len(text.split('Z')[0]) >= 14):
        return value.replace(b',', b'.')
    date = datetime.datetime(year, month, day, hour, minute) + datetime.timedelta(seconds=second)
    if zone != 'Z':
        delta = datetime.timedelta(hours=int(zone[1:3]), minutes=int(zone[3:5]))
        date += delta if zone[0] == '-' else -delta
    result = date.strftime('%y%m%d%H%M%S' if kind == 23 else '%Y%m%d%H%M%S')
    if fraction:
        result += '.' + fraction
    return (result + 'Z').encode('ascii')


def adjudicate(data):
    reader = Reader(data)
    root = reader.read(0)
    rejection = appliance_rejection(Reader(data, appliance=True))
    canonical = Canonicalizer(reader).encode(root)
    return canonical, rejection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, type=Path)
    args = parser.parse_args()
    # BER can have considerably deeper nesting than typical DER certificates.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    for path in sorted(args.corpus.glob('*.der'), key=lambda item: item.name):
        if not path.is_file():
            continue
        canonical, rejection = adjudicate(path.read_bytes())
        record = {
            'object': path.name,
            'canonical_der': base64.b64encode(canonical).decode('ascii'),
            'verdict': 'accept' if rejection is None else 'reject',
            'first_rejecting_element': rejection,
        }
        print(json.dumps(record, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
