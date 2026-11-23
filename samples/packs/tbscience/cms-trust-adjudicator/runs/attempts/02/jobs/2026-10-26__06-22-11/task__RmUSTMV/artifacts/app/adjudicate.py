#!/usr/bin/env python3
"""Canonicalize BER CMS SignedData and reproduce the appliance's TLV policy.

The two operations deliberately do not share a definition of 'canonical':
DER SET OF ordering uses canonical member encodings, whereas the appliance's
certificate reader compares the original member encodings.  The latter also
has different rules for CMS, certificates, and encapsulated extension values.
All offsets in the policy are positions in the original byte stream.
"""

import argparse
import base64
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys


def der_length(length):
    if length < 128:
        return bytes((length,))
    value = length.to_bytes((length.bit_length() + 7) // 8, 'big')
    return bytes((128 | len(value),)) + value


def der_tag(tag_class, constructed, number):
    prefix = (tag_class << 6) | (32 if constructed else 0)
    if number < 31:
        return bytes((prefix | number,))
    parts = [number & 127]
    number >>= 7
    while number:
        parts.append(128 | (number & 127))
        number >>= 7
    return bytes((prefix | 31,)) + bytes(reversed(parts))


def minimal_integer(value):
    p = 0
    while p + 1 < len(value):
        if value[p] == 0 and value[p + 1] < 128:
            p += 1
        elif value[p] == 255 and value[p + 1] >= 128:
            p += 1
        else:
            break
    return value[p:]


def canonical_time(tag, value):
    """Normalize the zoned ASN.1 time forms used by certificates and CMS.

    Fractional hours/minutes are expanded using integer arithmetic, preserving
    arbitrarily precise fractions without rounding through floating point.
    """
    pattern = (rb'(\d{10}(?:\d{2})?)(Z|[+-]\d{4})' if tag == 23 else
               rb'(\d{10}(?:\d{2})?(?:\d{2})?)(?:[.,](\d+))?(Z|[+-]\d{2}(?:\d{2})?)')
    match = re.fullmatch(pattern, value)
    if match is None:
        return value
    digits = match[1].decode('ascii')
    zone = match[2] if tag == 23 else match[3]
    if tag == 23:
        year = int(digits[:2])
        year += 1900 if year >= 50 else 2000
        rest = digits[2:]
        fraction = None
    else:
        year = int(digits[:4])
        rest = digits[4:]
        fraction = match[2]
    month, day, hour = int(rest[:2]), int(rest[2:4]), int(rest[4:6])
    minute = int(rest[6:8]) if len(rest) >= 8 else 0
    second = int(rest[8:10]) if len(rest) >= 10 else 0
    # datetime has no leap-second representation. Already canonical leap
    # seconds retain their exact representation.
    if second == 60:
        return value
    instant = datetime(year, month, day, hour, minute, second)
    tail = ''
    if fraction:
        units = 3600 if len(rest) == 6 else 60 if len(rest) == 8 else 1
        quotient, remainder = divmod(int(fraction) * units, 10 ** len(fraction))
        instant += timedelta(seconds=quotient)
        if remainder:
            tail = '.' + str(remainder).zfill(len(fraction)).rstrip('0')
    if zone != b'Z':
        minutes = int(zone[1:3]) * 60 + (int(zone[3:5]) if len(zone) == 5 else 0)
        instant -= timedelta(minutes=minutes if zone[:1] == b'+' else -minutes)
    year_text = f'{instant.year % 100:02d}' if tag == 23 else f'{instant.year:04d}'
    return (year_text + f'{instant.month:02d}{instant.day:02d}'
            f'{instant.hour:02d}{instant.minute:02d}{instant.second:02d}'
            + tail + 'Z').encode('ascii')


class EncodingError(ValueError):
    pass


class Element:
    __slots__ = ('start', 'content', 'stop', 'end', 'tag_class', 'tag',
                 'constructed', 'indefinite', 'minimal_length')

    def is_tag(self, number, tag_class=0):
        return self.tag_class == tag_class and self.tag == number


class Reader:
    """A byte-addressed TLV reader; primitive contents remain opaque.

    Indefinite lengths are resolved with a stack, rather than Python recursion.
    Cached element bounds also serve the independent appliance traversal, which
    can enter an OCTET STRING without changing its meaning for DER encoding.
    """

    def __init__(self, data, appliance_limits=False):
        self.data = data
        self.cache = {}
        self.appliance_limits = appliance_limits

    def _header(self, start):
        data = self.data
        n = len(data)
        if start < 0 or start >= n:
            raise EncodingError('missing tag')
        e = Element()
        e.start = start
        p = start
        first = data[p]
        p += 1
        e.tag_class = first >> 6
        e.constructed = bool(first & 32)
        e.tag = first & 31
        if e.tag == 31:
            e.tag = 0
            tag_octets = 0
            while True:
                if p >= n:
                    raise EncodingError('truncated tag')
                if self.appliance_limits and tag_octets == 6:
                    raise EncodingError('tag exceeds appliance reader limit')
                octet = data[p]
                p += 1
                tag_octets += 1
                e.tag = (e.tag << 7) | (octet & 127)
                if not octet & 128:
                    break
        if p >= n:
            raise EncodingError('missing length')
        first = data[p]
        p += 1
        e.indefinite = first == 128
        e.minimal_length = True
        if e.indefinite:
            e.content = p
            e.stop = e.end = None
            return e
        if first & 128:
            count = first & 127
            if self.appliance_limits and count > 8:
                raise EncodingError('length exceeds appliance reader limit')
            if p + count > n:
                raise EncodingError('truncated length')
            length = int.from_bytes(data[p:p + count], 'big')
            e.minimal_length = length >= 128 and data[p] != 0
            p += count
        else:
            length = first
        e.content = p
        e.stop = e.end = p + length
        if e.end > n:
            raise EncodingError('truncated value')
        return e

    def read(self, start):
        if start in self.cache:
            return self.cache[start]
        root = self._header(start)
        if not root.indefinite:
            self.cache[start] = root
            return root
        pending = [root]
        p = root.content
        while pending:
            if p + 2 > len(self.data):
                raise EncodingError('missing end-of-contents')
            if self.data[p:p + 2] == b'\x00\x00':
                e = pending.pop()
                e.stop = p
                e.end = p + 2
                self.cache[e.start] = e
                p += 2
            else:
                e = self.cache.get(p)
                if e is None:
                    e = self._header(p)
                if e.end is None:
                    pending.append(e)
                    p = e.content
                else:
                    self.cache[p] = e
                    p = e.end
        return root

    def value(self, e):
        return self.data[e.content:e.stop]

    def raw(self, e):
        return self.data[e.start:e.end]

    def children(self, e):
        """The appliance stops at EOC or an unreadable child."""
        p = e.content
        while p < e.stop:
            try:
                child = self.read(p)
            except EncodingError:
                return
            if child.is_tag(0) and not child.constructed and child.stop == child.content:
                return
            yield child
            p = child.end


class AppliancePolicy:
    """Policy recovered from the appliance, independent of certificate identity.

    Certificate ranges include their headers, but not the CertificateSet header.
    Extension payload ranges exclude the extnValue header. The appliance finds
    them structurally, without consulting an OID or the trust store. Its fixed
    range tables hold sixteen certificate choices and 128 extension values.
    """

    def __init__(self, reader, root):
        self.reader = reader
        self.certificates = []
        self.payloads = []
        self.first = None
        self._locate_ranges(root)

    @staticmethod
    def contains(ranges, position):
        return any(start <= position < end for start, end in ranges)

    def _locate_ranges(self, root):
        r = self.reader
        if not root.constructed:
            return
        outer = list(r.children(root))
        if len(outer) < 2 or not outer[1].constructed:
            return
        explicit = list(r.children(outer[1]))
        if not explicit or not explicit[0].constructed:
            return
        for field in r.children(explicit[0]):
            if field.is_tag(0, 2) and field.constructed:
                for cert in r.children(field):
                    if len(self.certificates) >= 16:
                        break
                    self.certificates.append((cert.start, cert.end))
            if len(self.certificates) >= 16:
                break
        for start, _ in self.certificates:
            cert = r.read(start)
            if not cert.constructed:
                continue
            fields = list(r.children(cert))
            if not fields or not fields[0].constructed:
                continue
            for field in r.children(fields[0]):
                if not (field.is_tag(3, 2) and field.constructed):
                    continue
                wrapped = list(r.children(field))
                if not wrapped:
                    break
                for extension in r.children(wrapped[0]):
                    if len(self.payloads) >= 128:
                        break
                    fields = list(r.children(extension))
                    if fields and fields[-1].is_tag(4):
                        value = fields[-1]
                        self.payloads.append((value.content, value.stop))

    def reject(self, e):
        if self.first is None or e.start < self.first:
            self.first = e.start

    def check(self, e, parent, index):
        r = self.reader
        strict = (self.contains(self.payloads, e.start)
                  or not self.contains(self.certificates, e.start))
        if not strict:
            if e.is_tag(17) and e.constructed:
                previous = None
                for child in r.children(e):
                    raw = r.raw(child)
                    # The device requires strictly increasing *original* TLVs,
                    # including their original tag and length octets.
                    if previous is not None and previous >= raw:
                        self.reject(e)
                        break
                    previous = raw
            return
        # Indefinite lengths are accepted; redundant definite lengths are not.
        if not e.minimal_length:
            self.reject(e)
        if e.tag_class != 0:
            return
        if e.constructed:
            if e.tag in (3, 4):
                self.reject(e)
            return
        value = r.value(e)
        if e.tag == 2:
            if not value or value != minimal_integer(value):
                self.reject(e)
        elif (e.tag == 1 and parent is not None and parent.is_tag(16)
              and index == 0 and value == b'\x00'):
            self.reject(e)

    def run(self, root):
        r = self.reader
        pending = [(root, None, 0, False)]
        while pending:
            e, parent, index, is_child = pending.pop()
            self.check(e, parent, index)
            if e.constructed:
                children = list(r.children(e))
                pending.extend((c, e, i, True)
                               for i, c in reversed(list(enumerate(children))))
            elif is_child and e.is_tag(4) and e.stop > e.content:
                if (self.contains(self.payloads, e.content)
                        and self.contains(self.payloads, e.stop - 1)):
                    # The reader enters only the first TLV of a primitive
                    # extension OCTET STRING, and repeats for nested octets.
                    try:
                        inner = r.read(e.content)
                    except EncodingError:
                        continue
                    pending.append((inner, None, 0, False))
        return self.first


class CanonicalDER:
    """DER encoding with schema annotations for CMS and X.509 wrappers.

    OCTET STRING bytes, including encoded extension values and signatures, are
    the value of that string, not children of it. Only constructed string
    fragments are flattened. In particular the policy's inspection of an
    extension's contents must not silently rewrite those octets.
    """

    SIGNED_DATA = bytes.fromhex('2a864886f70d010702')
    COUNTER_SIGNATURE = bytes.fromhex('2a864886f70d010906')
    PSS = bytes.fromhex('2a864886f70d01010a')
    OAEP = bytes.fromhex('2a864886f70d010107')

    def __init__(self, reader, root):
        self.reader = reader
        self.root = root
        self.nodes = []
        self.children = {}
        self.set_of = set()
        self.omit = set()
        self.base_type = {}
        pending = [root]
        while pending:
            e = pending.pop()
            self.nodes.append(e)
            children = list(reader.children(e)) if e.constructed else []
            self.children[e.start] = children
            pending.extend(reversed(children))
            if e.is_tag(17):
                self.set_of.add(e.start)
        self._schema()

    def kids(self, e):
        return self.children.get(e.start, [])

    def _integer_is(self, e, value):
        return (e.is_tag(2) and not e.constructed
                and bool(self.reader.value(e))
                and int.from_bytes(self.reader.value(e), 'big', signed=True) == value)

    def _extensions(self, wrapper):
        for extension in self.kids(wrapper):
            fields = self.kids(extension)
            if (len(fields) >= 3 and fields[1].is_tag(1)
                    and self.reader.value(fields[1]) == b'\x00'):
                self.omit.add(fields[1].start)

    def _certificate(self, certificate):
        fields = self.kids(certificate)
        if not fields:
            return
        tbs = fields[0]
        fields = self.kids(tbs)
        if fields and fields[0].is_tag(0, 2):
            version = self.kids(fields[0])
            if len(version) == 1 and self._integer_is(version[0], 0):
                self.omit.add(fields[0].start)
        for field in fields:
            if field.is_tag(3, 2):
                for extensions in self.kids(field):
                    self._extensions(extensions)
            elif field.tag_class == 2 and field.tag in (1, 2):
                self.base_type[field.start] = 3  # IMPLICIT UniqueIdentifier

    def _crl(self, crl):
        fields = self.kids(crl)
        if not fields:
            return
        for field in self.kids(fields[0]):
            if field.is_tag(0, 2):
                for extensions in self.kids(field):
                    self._extensions(extensions)
            elif field.is_tag(16):
                # A revokedCertificates SEQUENCE contains entry SEQUENCEs.
                for entry in self.kids(field):
                    items = self.kids(entry)
                    if len(items) >= 3 and items[0].is_tag(2):
                        self._extensions(items[2])

    def _signer(self, signer):
        fields = self.kids(signer)
        if len(fields) > 1 and fields[1].is_tag(0, 2):
            self.base_type[fields[1].start] = 4  # SubjectKeyIdentifier
        for field in fields[2:]:
            if field.tag_class == 2 and field.tag in (0, 1):
                self.set_of.add(field.start)

    def _signed_data(self, sd):
        fields = self.kids(sd)
        for field in fields[3:]:
            if field.is_tag(0, 2):
                self.set_of.add(field.start)
                for certificate in self.kids(field):
                    if certificate.is_tag(16):
                        self._certificate(certificate)
            elif field.is_tag(1, 2):
                self.set_of.add(field.start)
                for crl in self.kids(field):
                    if crl.is_tag(16):
                        self._crl(crl)
            elif field.is_tag(17):
                for signer in self.kids(field):
                    self._signer(signer)

    def _schema(self):
        r = self.reader
        for e in self.nodes:
            fields = self.kids(e)
            if not (e.is_tag(16) and fields and fields[0].is_tag(6)):
                continue
            oid = r.value(fields[0])
            if oid == self.SIGNED_DATA and len(fields) >= 2:
                for sd in self.kids(fields[1]):
                    self._signed_data(sd)
            elif oid == self.COUNTER_SIGNATURE and len(fields) >= 2:
                for signer in self.kids(fields[1]):
                    self._signer(signer)

    @staticmethod
    def _oid(value):
        """Minimal base-128 subidentifiers (also works for RELATIVE-OID)."""
        out = bytearray()
        number = 0
        for byte in value:
            number = (number << 7) | (byte & 127)
            if not byte & 128:
                digits = [number & 127]
                number >>= 7
                while number:
                    digits.append(128 | (number & 127))
                    number >>= 7
                out.extend(reversed(digits))
        return bytes(out)

    def encode(self):
        r = self.reader
        encoded = {}
        strings = {}
        # Explicit algorithm defaults are tested after their children have
        # themselves been canonicalized.
        sha1 = bytes.fromhex('300906052b0e03021a0500')
        mgf1 = bytes.fromhex('301606092a864886f70d010108') + sha1
        pspecified = bytes.fromhex('300d06092a864886f70d0101090400')
        parameter_defaults = {}
        for e in self.nodes:
            fields = self.kids(e)
            if (e.is_tag(16) and len(fields) == 2
                    and fields[0].is_tag(6) and fields[1].is_tag(16)):
                oid = r.value(fields[0])
                if oid == self.PSS:
                    parameter_defaults[fields[1].start] = {
                        0: sha1, 1: mgf1, 2: b'\x02\x01\x14', 3: b'\x02\x01\x01'}
                elif oid == self.OAEP:
                    parameter_defaults[fields[1].start] = {
                        0: sha1, 1: mgf1, 2: pspecified}
        for e in reversed(self.nodes):
            tag = self.base_type.get(e.start, e.tag if e.tag_class == 0 else None)
            children = self.kids(e)
            constructed = e.constructed
            if tag in (3, 4) or (e.tag_class == 0 and tag in
                                  (7, 12, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30)):
                if e.constructed:
                    parts = [strings[c.start] for c in children]
                    if tag == 3:
                        value = (parts[-1][:1] if parts else b'\x00')
                        value += b''.join(part[1:] for part in parts)
                    else:
                        value = b''.join(parts)
                else:
                    value = r.value(e)
                if tag == 3 and value and value[0] and len(value) > 1:
                    value = value[:-1] + bytes((value[-1] & (255 << value[0]) & 255,))
                strings[e.start] = value
                constructed = False
            elif e.constructed:
                defaults = parameter_defaults.get(e.start, {})
                pieces = []
                for c in children:
                    if c.start in self.omit:
                        continue
                    grandchildren = self.kids(c)
                    if (c.tag_class == 2 and c.tag in defaults
                            and len(grandchildren) == 1
                            and encoded[grandchildren[0].start] == defaults[c.tag]):
                        continue
                    pieces.append(encoded[c.start])
                if e.start in self.set_of:
                    pieces.sort()
                value = b''.join(pieces)
            else:
                value = r.value(e)
                if e.tag_class == 0:
                    if e.tag in (2, 10):
                        value = minimal_integer(value)
                    elif e.tag == 1:
                        value = b'\xff' if any(value) else b'\x00'
                    elif e.tag in (6, 13):
                        value = self._oid(value)
                    elif e.tag in (23, 24):
                        value = canonical_time(e.tag, value)
            encoded[e.start] = (der_tag(e.tag_class, constructed, e.tag)
                                + der_length(len(value)) + value)
        return encoded[self.root.start]


def adjudicate(data):
    reader = Reader(data)
    root = reader.read(0)
    # Reader failures are not rejection events in this appliance. Keep its
    # machine-word header limits out of the general BER-to-DER reader.
    policy_reader = Reader(data, appliance_limits=True)
    try:
        policy_root = policy_reader.read(0)
    except EncodingError:
        first = None
    else:
        first = AppliancePolicy(policy_reader, policy_root).run(policy_root)
    canonical = CanonicalDER(reader, root).encode()
    return canonical, first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, type=Path)
    args = parser.parse_args()
    for path in sorted(args.corpus.glob('*.der'), key=lambda p: p.name):
        if not path.is_file():
            continue
        canonical, first = adjudicate(path.read_bytes())
        record = {
            'object': path.name,
            'canonical_der': base64.b64encode(canonical).decode('ascii'),
            'verdict': 'accept' if first is None else 'reject',
            'first_rejecting_element': first,
        }
        sys.stdout.write(json.dumps(record, separators=(',', ':')) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
