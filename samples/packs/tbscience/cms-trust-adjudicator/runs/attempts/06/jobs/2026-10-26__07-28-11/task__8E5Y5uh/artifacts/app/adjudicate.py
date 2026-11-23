#!/usr/bin/env python3
"""Canonicalize CMS SignedData and reproduce the appliance's encoding policy.

The two operations deliberately use different rules.  DER is a transformation
of the ASN.1 value; the appliance tests particular properties of the original
encoding, using original byte offsets.  In particular its certificate SET
comparison uses input encodings, not canonical encodings.

Only Python's standard library is needed.  Certificate signatures and trust
paths have no bearing on this appliance's encoding-policy decision.
"""

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys


class ASN1Error(ValueError):
    pass


@dataclass(frozen=True)
class Element:
    start: int
    cls: int
    tag: int
    constructed: bool
    value_start: int
    value_end: int
    end: int
    indefinite: bool
    minimal_length: bool

    def is_tag(self, tag, cls=0):
        return self.cls == cls and self.tag == tag


class BER:
    """Read TLVs without interpreting primitive contents as more TLVs.

    The appliance reader is deliberately less validating than an ASN.1 schema
    decoder.  Its walks stop at unreadable data (notably in extension octets).
    Its header-size limits are reproduced only in appliance mode.
    """

    def __init__(self, data):
        self.data = data
        self.cache = {}

    def read(self, offset=0, appliance=False):
        key = (offset, appliance)
        if key in self.cache:
            return self.cache[key]
        data = self.data
        size = len(data)
        p = offset
        if not 0 <= p < size:
            raise ASN1Error('missing tag')
        first = data[p]
        p += 1
        cls, constructed, tag = first >> 6, bool(first & 32), first & 31
        if tag == 31:
            tag = 0
            count = 0
            while True:
                if p >= size or (appliance and count == 6):
                    raise ASN1Error('invalid high tag')
                b = data[p]
                p += 1
                count += 1
                tag = (tag << 7) | (b & 127)
                if not b & 128:
                    break
        if p >= size:
            raise ASN1Error('missing length')
        b = data[p]
        p += 1
        indefinite = b == 128
        minimal = True
        if indefinite:
            value_start = p
            while True:
                if p + 2 > size:
                    raise ASN1Error('unterminated indefinite value')
                if data[p:p + 2] == b'\x00\x00':
                    break
                p = self.read(p, appliance).end
            value_end, end = p, p + 2
        else:
            if b & 128:
                count = b & 127
                if p + count > size or (appliance and count > 8):
                    raise ASN1Error('invalid long length')
                length = int.from_bytes(data[p:p + count], 'big')
                minimal = length > 127 and data[p] != 0
                p += count
            else:
                length = b
            value_start = p
            value_end = end = p + length
            if end > size:
                raise ASN1Error('truncated content')
        node = Element(offset, cls, tag, constructed, value_start, value_end,
                       end, indefinite, minimal)
        self.cache[key] = node
        return node

    def contents(self, node):
        return self.data[node.value_start:node.value_end]

    def raw(self, node):
        return self.data[node.start:node.end]

    def children(self, node, appliance=False):
        p = node.value_start
        while p < node.value_end:
            try:
                child = self.read(p, appliance)
            except ASN1Error:
                if appliance:
                    return
                raise
            if (child.is_tag(0) and not child.constructed
                    and child.value_start == child.value_end):
                return
            if not appliance and child.end > node.value_end:
                raise ASN1Error('child exceeds parent')
            yield child
            p = child.end

    def child(self, node, index, appliance=False):
        for i, child in enumerate(self.children(node, appliance)):
            if i == index:
                return child
        return None


def inside(offset, ranges):
    return any(start <= offset < end for start, end in ranges)


class Appliance:
    """The appliance's region discovery and original-encoding checks.

    The permissive regions are up to sixteen certificate choices immediately
    inside SignedData.certificates.  Within them the contents of up to 128
    extnValue OCTET STRINGs take precedence and use the strict rules.  The
    enclosing CertificateSet and extnValue headers are outside those regions.
    These limits and the raw, strictly-increasing SET test belong to the
    appliance, not to DER.
    """

    def __init__(self, ber):
        self.ber = ber
        self.certificates = []
        self.payloads = []
        self.first = None

    def discover(self, root):
        ber = self.ber
        if not root.constructed:
            return
        wrapper = ber.child(root, 1, True)
        if wrapper is None or not wrapper.constructed:
            return
        signed = ber.child(wrapper, 0, True)
        if signed is None or not signed.constructed:
            return
        certs = []
        for field in ber.children(signed, True):
            if field.is_tag(0, 2) and field.constructed:
                for cert in ber.children(field, True):
                    if len(certs) == 16:
                        break
                    certs.append(cert)
                    self.certificates.append((cert.start, cert.end))
            if len(certs) == 16:
                break
        for cert in certs:
            if not cert.constructed:
                continue
            tbs = ber.child(cert, 0, True)
            if tbs is None or not tbs.constructed:
                continue
            for field in ber.children(tbs, True):
                if not field.is_tag(3, 2) or not field.constructed:
                    continue
                extensions = ber.child(field, 0, True)
                if extensions is None:
                    break
                for extension in ber.children(extensions, True):
                    if len(self.payloads) == 128:
                        break
                    last = None
                    for last in ber.children(extension, True):
                        pass
                    if last is not None and last.is_tag(4):
                        self.payloads.append((last.value_start, last.value_end))

    def reject(self, node):
        if self.first is None or node.start < self.first:
            self.first = node.start

    def check(self, node, parent_tag=None, index=0):
        ber = self.ber
        if (not inside(node.start, self.payloads)
                and inside(node.start, self.certificates)):
            if node.is_tag(17) and node.constructed:
                previous = None
                for child in ber.children(node, True):
                    raw = ber.raw(child)
                    if previous is not None and previous >= raw:
                        self.reject(node)
                        break
                    previous = raw
            return
        if not node.minimal_length:
            self.reject(node)
        if node.cls != 0:
            return
        if node.constructed:
            if node.tag in (3, 4):
                self.reject(node)
            return
        value = ber.contents(node)
        if node.tag == 2:
            if (not value or (len(value) > 1
                    and ((value[0] == 0 and value[1] < 128)
                         or (value[0] == 255 and value[1] >= 128)))):
                self.reject(node)
        elif (node.tag == 1 and parent_tag == 16 and index == 0
              and value == b'\x00'):
            self.reject(node)

    def walk_root(self, offset):
        try:
            node = self.ber.read(offset, True)
        except ASN1Error:
            return
        self.check(node)
        if node.constructed:
            self.walk_children(node)

    def walk_children(self, parent):
        parent_tag = parent.tag if parent.cls == 0 else None
        for i, node in enumerate(self.ber.children(parent, True)):
            self.check(node, parent_tag, i)
            if node.constructed:
                self.walk_children(node)
            elif (node.is_tag(4) and node.value_start < node.value_end
                  and inside(node.value_start, self.payloads)
                  and inside(node.value_end - 1, self.payloads)):
                # The appliance enters an encapsulated value at its original
                # location.  It does not concatenate fragments before checking.
                self.walk_root(node.value_start)

    def evaluate(self):
        try:
            self.discover(self.ber.read(0, True))
        except ASN1Error:
            pass
        self.walk_root(0)
        return self.first


def der_length(length):
    if length < 128:
        return bytes([length])
    value = length.to_bytes((length.bit_length() + 7) // 8, 'big')
    return bytes([128 | len(value)]) + value


def base128(value):
    result = [value & 127]
    value >>= 7
    while value:
        result.append(128 | (value & 127))
        value >>= 7
    return bytes(reversed(result))


def der_tag(cls, constructed, tag):
    first = (cls << 6) | (32 if constructed else 0)
    if tag < 31:
        return bytes([first | tag])
    return bytes([first | 31]) + base128(tag)


def tlv(cls, constructed, tag, value):
    return der_tag(cls, constructed, tag) + der_length(len(value)) + value


def minimal_integer(value):
    p = 0
    while p + 1 < len(value):
        if ((value[p] == 0 and value[p + 1] < 128)
                or (value[p] == 255 and value[p + 1] >= 128)):
            p += 1
        else:
            break
    return value[p:]


def oid_value(value):
    """Decode the subidentifiers, including the combined first two arcs."""
    arcs = []
    current = 0
    for b in value:
        current = (current << 7) | (b & 127)
        if not b & 128:
            arcs.append(current)
            current = 0
    if value and value[-1] & 128:
        raise ASN1Error('unterminated OID')
    return tuple(arcs)


def oid_bytes(dotted):
    arcs = tuple(map(int, dotted.split('.')))
    return b''.join(base128(n) for n in (40 * arcs[0] + arcs[1],) + arcs[2:])


SIGNED_DATA = oid_bytes('1.2.840.113549.1.7.2')
COUNTER_SIGNATURE = oid_bytes('1.2.840.113549.1.9.6')
TIMESTAMP_TOKEN = oid_bytes('1.2.840.113549.1.9.16.2.14')
SIGNING_CERTIFICATE_V2 = oid_bytes('1.2.840.113549.1.9.16.2.47')
RSA_PSS = oid_bytes('1.2.840.113549.1.1.10')
RSA_OAEP = oid_bytes('1.2.840.113549.1.1.7')
MGF1 = oid_bytes('1.2.840.113549.1.1.8')
P_SPECIFIED = oid_bytes('1.2.840.113549.1.1.9')
SHA1 = oid_bytes('1.3.14.3.2.26')
SHA256 = oid_bytes('2.16.840.1.101.3.4.2.1')


def canonical_time(value, tag):
    """Normalize the timezone and precision without losing fractional digits."""
    text = value.decode('ascii')
    if tag == 23:
        match = re.fullmatch(r'(\d{10})(\d{2})?(Z|[+-]\d{4})', text)
        if match is None:
            raise ASN1Error('invalid UTCTime')
        digits, seconds, zone = match.groups()
        year = int(digits[:2])
        year += 2000 if year < 50 else 1900
        stamp = datetime(year, int(digits[2:4]), int(digits[4:6]),
                         int(digits[6:8]), int(digits[8:10]),
                         int(seconds or '0'))
        fraction = ''
    else:
        match = re.fullmatch(
            r'(\d{10})(\d{2})?(\d{2})?(?:[.,](\d+))?(Z|[+-]\d{4})', text)
        if match is None:
            raise ASN1Error('invalid GeneralizedTime')
        digits, minutes, seconds, decimal, zone = match.groups()
        stamp = datetime(int(digits[:4]), int(digits[4:6]), int(digits[6:8]),
                         int(digits[8:10]), int(minutes or '0'),
                         int(seconds or '0'))
        fraction = ''
        if decimal:
            unit = 1 if seconds is not None else 60 if minutes is not None else 3600
            whole, remainder = divmod(int(decimal) * unit, 10 ** len(decimal))
            stamp += timedelta(seconds=whole)
            fraction = str(remainder).zfill(len(decimal)).rstrip('0')
    if zone != 'Z':
        offset = timedelta(hours=int(zone[1:3]), minutes=int(zone[3:5]))
        stamp += -offset if zone[0] == '+' else offset
    year = f'{stamp.year % 100:02d}' if tag == 23 else f'{stamp.year:04d}'
    result = year + stamp.strftime('%m%d%H%M%S')
    if fraction:
        result += '.' + fraction
    return (result + 'Z').encode('ascii')


class CanonicalDER:
    """Schema annotations plus a bottom-up DER encoder.

    Context tags need schema information: certificates and attributes are
    implicitly tagged SET OFs, while a version or ContentInfo wrapper is
    EXPLICIT.  Primitive OCTET STRINGs (including extnValue) and BIT STRINGs
    are bytes/bits, not invitations to rewrite embedded ASN.1.  Changing those
    octets would change the abstract value of the enclosing CMS object.
    """

    STRING_TAGS = frozenset((4, 7, 12, 18, 19, 20, 21, 22, 23, 24,
                             25, 26, 27, 28, 30))

    def __init__(self, ber):
        self.ber = ber
        self.sets = set()
        self.omit = set()
        self.implicit_strings = {}
        self.encoded = {}

    def children(self, node):
        return list(self.ber.children(node))

    def oid(self, node):
        if not node.is_tag(6):
            return None
        return b''.join(base128(a) for a in oid_value(self.ber.contents(node)))

    def annotate(self, root):
        if not root.is_tag(16) or not root.constructed:
            return
        fields = self.children(root)
        if (len(fields) >= 2
                and self.oid(fields[0]) == SIGNED_DATA
                and fields[1].is_tag(0, 2)):
            signed = self.ber.child(fields[1], 0)
            if signed is not None:
                self.signed_data(signed)

    def signed_data(self, node):
        fields = self.children(node)
        if len(fields) < 4:
            return
        for algorithm in self.children(fields[1]):
            self.algorithm(algorithm)
        for field in fields[3:]:
            if field.is_tag(0, 2):
                self.sets.add(field.start)
                for certificate in self.children(field):
                    if certificate.is_tag(16):
                        self.certificate(certificate)
            elif field.is_tag(1, 2):
                self.sets.add(field.start)
                for crl in self.children(field):
                    if crl.is_tag(16):
                        self.crl(crl)
            elif field.is_tag(17):
                for signer in self.children(field):
                    self.signer_info(signer)

    def certificate(self, node):
        tbs = self.ber.child(node, 0)
        if tbs is None:
            return
        fields = self.children(tbs)
        version_present = bool(fields and fields[0].is_tag(0, 2))
        if version_present:
            version = self.ber.child(fields[0], 0)
            if (version is not None and version.is_tag(2)
                    and int.from_bytes(self.ber.contents(version), 'big',
                                       signed=True) == 0):
                self.omit.add(fields[0].start)
        signature_index = 2 if version_present else 1
        if len(fields) > signature_index:
            self.algorithm(fields[signature_index])
        spki_index = 6 if version_present else 5
        if len(fields) > spki_index:
            public_key_algorithm = self.ber.child(fields[spki_index], 0)
            if public_key_algorithm is not None:
                self.algorithm(public_key_algorithm)
        outer_algorithm = self.ber.child(node, 1)
        if outer_algorithm is not None:
            self.algorithm(outer_algorithm)
        for field in fields:
            if field.is_tag(3, 2):
                extensions = self.ber.child(field, 0)
                if extensions is not None:
                    self.extensions(extensions)
            elif field.cls == 2 and field.tag in (1, 2):
                self.implicit_strings[field.start] = 3

    def extensions(self, node):
        for extension in self.children(node):
            fields = self.children(extension)
            if (len(fields) >= 3 and fields[1].is_tag(1)
                    and not any(self.ber.contents(fields[1]))):
                self.omit.add(fields[1].start)

    def crl(self, node):
        tbs = self.ber.child(node, 0)
        if tbs is None:
            return
        fields = self.children(tbs)
        signature_index = 1 if fields and fields[0].is_tag(2) else 0
        if len(fields) > signature_index:
            self.algorithm(fields[signature_index])
        outer_algorithm = self.ber.child(node, 1)
        if outer_algorithm is not None:
            self.algorithm(outer_algorithm)
        for field in fields:
            if field.is_tag(0, 2):
                extensions = self.ber.child(field, 0)
                if extensions is not None:
                    self.extensions(extensions)
        # revokedCertificates follows thisUpdate and optional nextUpdate.
        i = 1 if fields and fields[0].is_tag(2) else 0
        i += 3  # signature, issuer, thisUpdate
        if i < len(fields) and fields[i].cls == 0 and fields[i].tag in (23, 24):
            i += 1
        if i < len(fields) and fields[i].is_tag(16):
            for entry in self.children(fields[i]):
                entry_fields = self.children(entry)
                if len(entry_fields) > 2:
                    self.extensions(entry_fields[2])

    def signer_info(self, node):
        fields = self.children(node)
        if len(fields) > 1 and fields[1].is_tag(0, 2):
            self.implicit_strings[fields[1].start] = 4
        if len(fields) > 2:
            self.algorithm(fields[2])
        signature_index = 4 if len(fields) > 3 and fields[3].is_tag(0, 2) else 3
        if len(fields) > signature_index:
            self.algorithm(fields[signature_index])
        for field in fields[3:]:
            if field.cls == 2 and field.tag in (0, 1):
                self.attributes(field)

    def attributes(self, node):
        self.sets.add(node.start)
        for attr in self.children(node):
            fields = self.children(attr)
            if len(fields) < 2:
                continue
            oid = self.oid(fields[0])
            if oid == COUNTER_SIGNATURE:
                for signer in self.children(fields[1]):
                    self.signer_info(signer)
            elif oid == TIMESTAMP_TOKEN:
                for token in self.children(fields[1]):
                    self.annotate(token)
            elif oid == SIGNING_CERTIFICATE_V2:
                for value in self.children(fields[1]):
                    certs = self.ber.child(value, 0)
                    if certs is not None:
                        for cert in self.children(certs):
                            algorithm = self.ber.child(cert, 0)
                            if algorithm is not None and algorithm.is_tag(16):
                                parts = self.children(algorithm)
                                if len(parts) == 1 and self.oid(parts[0]) == SHA256:
                                    self.omit.add(algorithm.start)
                                else:
                                    self.algorithm(algorithm)

    def algorithm(self, node):
        """DEFAULT components in standard RSA algorithm parameter schemas.

        Equality here is ASN.1 value equality, not cryptographic equivalence:
        absent AlgorithmIdentifier parameters and a present NULL are distinct.
        """
        if not node.constructed:
            return
        fields = self.children(node)
        if len(fields) != 2:
            return
        oid = self.oid(fields[0])
        params = fields[1]
        if oid == MGF1:
            self.algorithm(params)
            return
        if oid not in (RSA_PSS, RSA_OAEP) or not params.is_tag(16):
            return
        for field in self.children(params):
            if field.cls != 2 or not field.constructed:
                continue
            components = self.children(field)
            if len(components) != 1:
                continue
            value = components[0]
            default = False
            if field.tag == 0:
                default = self.is_sha1_default(value)
                self.algorithm(value)
            elif field.tag == 1:
                parts = self.children(value) if value.constructed else []
                default = (len(parts) == 2 and self.oid(parts[0]) == MGF1
                           and self.is_sha1_default(parts[1]))
                self.algorithm(value)
            elif oid == RSA_PSS and field.tag in (2, 3) and value.is_tag(2):
                default = int.from_bytes(self.ber.contents(value), 'big', signed=True) == (
                    20 if field.tag == 2 else 1)
            elif oid == RSA_OAEP and field.tag == 2:
                parts = self.children(value) if value.constructed else []
                default = (len(parts) == 2 and self.oid(parts[0]) == P_SPECIFIED
                           and parts[1].is_tag(4)
                           and self.string_value(parts[1], 4) == b'')
            if default:
                self.omit.add(field.start)

    def is_sha1_default(self, node):
        parts = self.children(node) if node.constructed else []
        return (len(parts) == 2 and self.oid(parts[0]) == SHA1
                and parts[1].is_tag(5) and not self.ber.contents(parts[1]))

    def string_value(self, node, tag):
        if not node.constructed:
            value = self.ber.contents(node)
            if tag == 3:
                if not value or value[0] > 7:
                    raise ASN1Error('invalid BIT STRING')
                if len(value) == 1:
                    return b'\x00'
                unused = value[0]
                if unused:
                    value = value[:-1] + bytes([value[-1] & (255 << unused)])
            return value
        parts = []
        for child in self.children(node):
            if not child.is_tag(tag):
                raise ASN1Error('wrong string fragment type')
            parts.append(self.string_value(child, tag))
        if tag != 3:
            return b''.join(parts)
        if not parts:
            return b'\x00'
        # BER permits unused bits only in the final component.  All other
        # components contribute complete octets, even across nested fragments.
        if any(part[0] for part in parts[:-1]):
            raise ASN1Error('non-final BIT STRING fragment has unused bits')
        return parts[-1][:1] + b''.join(part[1:] for part in parts)

    def encode(self, node):
        if node.start in self.omit:
            return b''
        if node.start in self.encoded:
            return self.encoded[node.start]
        constructed = node.constructed
        tag = self.implicit_strings.get(node.start)
        if tag is None and node.cls == 0 and (node.tag == 3 or node.tag in self.STRING_TAGS):
            tag = node.tag
        if tag is not None:
            value = self.string_value(node, tag)
            if tag in (23, 24):
                value = canonical_time(value, tag)
            constructed = False
        elif constructed:
            children = [self.encode(child) for child in self.children(node)]
            if node.is_tag(17) or node.start in self.sets:
                children.sort()
            value = b''.join(children)
        else:
            value = self.ber.contents(node)
            if node.cls == 0:
                if node.tag in (2, 10):
                    value = minimal_integer(value)
                elif node.tag == 1:
                    value = b'\xff' if any(value) else b'\x00'
                elif node.tag in (6, 13):
                    value = b''.join(base128(a) for a in oid_value(value))
        result = tlv(node.cls, constructed, node.tag, value)
        self.encoded[node.start] = result
        return result

    def canonicalize(self):
        root = self.ber.read()
        self.annotate(root)
        return self.encode(root)


def adjudicate(data, name):
    ber = BER(data)
    first = Appliance(ber).evaluate()
    canonical = CanonicalDER(ber).canonicalize()
    return {
        'object': name,
        'canonical_der': base64.b64encode(canonical).decode('ascii'),
        'verdict': 'accept' if first is None else 'reject',
        'first_rejecting_element': first,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, type=Path)
    args = parser.parse_args()
    # Allow deeply nested legal BER constructions without imposing Python's
    # relatively small default recursion limit on the ASN.1 representation.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
    for path in sorted(args.corpus.glob('*.der'), key=lambda p: p.name):
        if path.is_file():
            record = adjudicate(path.read_bytes(), path.name)
            print(json.dumps(record, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
