#!/usr/bin/env python3
"""Canonicalize BER CMS objects and reproduce the appliance's acceptance policy.

Only the supplied input directory is read.  Acceptance is evaluated on original
byte offsets and encodings, independently of the DER transformation.
"""
import argparse
import base64
import json
from pathlib import Path


class EncodingError(ValueError):
    pass


class Element:
    __slots__ = ('start', 'content', 'stop', 'end', 'cls', 'tag', 'constructed',
                 'indefinite', 'minimal_length', 'children', 'role', 'omit',
                 'der', 'payload')

    def __init__(self, start, content, stop, end, cls, tag, constructed,
                 indefinite, minimal_length):
        self.start, self.content, self.stop, self.end = start, content, stop, end
        self.cls, self.tag, self.constructed = cls, tag, constructed
        self.indefinite, self.minimal_length = indefinite, minimal_length
        self.children = None
        self.role = None
        self.omit = False
        self.der = self.payload = None

    def is_tag(self, tag, cls=0):
        return self.cls == cls and self.tag == tag


def length_octets(n):
    if n < 128:
        return bytes((n,))
    b = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    return bytes((0x80 | len(b),)) + b


def tag_octets(cls, constructed, tag):
    prefix = cls | (32 if constructed else 0)
    if tag < 31:
        return bytes((prefix | tag,))
    out = [tag & 127]
    tag >>= 7
    while tag:
        out.append(128 | (tag & 127))
        tag >>= 7
    return bytes((prefix | 31,)) + bytes(reversed(out))


def tlv(cls, constructed, tag, value):
    return tag_octets(cls, constructed, tag) + length_octets(len(value)) + value


class Reader:
    """Offset-preserving TLV reader, including nested indefinite lengths."""
    def __init__(self, data, appliance=False):
        self.data = data
        self.cache = {}
        self.appliance = appliance

    def _header(self, start):
        data = self.data
        p = start
        if p >= len(data):
            raise EncodingError('missing tag')
        first = data[p]
        p += 1
        tag = first & 31
        if tag == 31:
            tag = 0
            groups = 0
            while True:
                if p >= len(data):
                    raise EncodingError('truncated tag')
                if self.appliance and groups == 6:
                    raise EncodingError('appliance tag limit')
                b = data[p]
                p += 1
                groups += 1
                tag = (tag << 7) | (b & 127)
                if b < 128:
                    break
        if p >= len(data):
            raise EncodingError('missing length')
        b = data[p]
        p += 1
        indefinite = b == 128
        minimal = True
        if b > 128:
            count = b & 127
            if self.appliance and count > 8:
                raise EncodingError('appliance length limit')
            if p + count > len(data):
                raise EncodingError('truncated length')
            size = int.from_bytes(data[p:p + count], 'big')
            minimal = size >= 128 and data[p] != 0
            p += count
        elif indefinite:
            size = 0
        else:
            size = b
        end = p + size
        if end > len(data):
            raise EncodingError('truncated value')
        return Element(start, p, end, end, first & 192, tag,
                       bool(first & 32), indefinite, minimal)

    def read(self, start):
        if start in self.cache:
            return self.cache[start]
        root = self._header(start)
        if not root.indefinite:
            self.cache[start] = root
            return root
        stack = [root]
        p = root.content
        while stack:
            if p + 2 > len(self.data):
                raise EncodingError('missing end of contents')
            if self.data[p:p + 2] == b'\x00\x00':
                node = stack.pop()
                node.stop, node.end = p, p + 2
                self.cache[node.start] = node
                p += 2
            else:
                node = self.cache.get(p)
                if node is not None:
                    p = node.end
                    continue
                node = self._header(p)
                if node.indefinite:
                    stack.append(node)
                    p = node.content
                else:
                    self.cache[p] = node
                    p = node.end
        return root

    def children(self, node):
        if node.children is None:
            result = []
            if node.constructed:
                p = node.content
                while p < node.stop:
                    child = self.read(p)
                    if child.is_tag(0) and not child.constructed and child.content == child.stop:
                        break
                    if child.end > node.stop:
                        raise EncodingError('child exceeds parent')
                    result.append(child)
                    p = child.end
            node.children = result
        return node.children

    def value(self, node):
        return self.data[node.content:node.stop]

    def raw(self, node):
        return self.data[node.start:node.end]


def appliance_members(reader, node):
    # The appliance stops at a parse failure, retaining the preceding members.
    # Its TLV reader bounds against the file, rather than the current container.
    result = []
    p = node.content
    while p < node.stop:
        try:
            child = reader.read(p)
        except EncodingError:
            break
        if child.is_tag(0) and not child.constructed and child.content == child.stop:
            break
        result.append(child)
        p = child.end
    return result


def appliance_ranges(reader, root):
    """Discover certificate regions and their extension-value exceptions.

    The appliance uses structural positions, not issuer identities or a trust
    database. Its region tables hold 16 certificates and 128 extension values.
    An extension's OCTET STRING header remains in the certificate region; its
    contents use the CMS checks, even for an unrecognized extension OID.
    """
    certificates, extensions = [], []
    try:
        if not root.constructed:
            return [], []
        wrapper = appliance_members(reader, root)[1]
        if not wrapper.constructed:
            return [], []
        signed_data = appliance_members(reader, wrapper)[0]
        if not signed_data.constructed:
            return [], []
        for field in appliance_members(reader, signed_data):
            if field.is_tag(0, 128) and field.constructed:
                for certificate in appliance_members(reader, field):
                    if len(certificates) == 16:
                        break
                    certificates.append(certificate)
            if len(certificates) == 16:
                break
    except (EncodingError, IndexError):
        pass
    for certificate in certificates:
        try:
            if not certificate.constructed:
                continue
            tbs = appliance_members(reader, certificate)[0]
            if not tbs.constructed:
                continue
            for field in appliance_members(reader, tbs):
                if not (field.is_tag(3, 128) and field.constructed):
                    continue
                sequence = appliance_members(reader, field)[0]
                for extension in appliance_members(reader, sequence):
                    if len(extensions) == 128:
                        break
                    fields = appliance_members(reader, extension)
                    if fields and fields[-1].is_tag(4):
                        value = fields[-1]
                        extensions.append((value.content, value.stop))
        except (EncodingError, IndexError):
            continue
    return [(n.start, n.end) for n in certificates], extensions


def appliance_rejection(reader, root):
    # Keep the appliance's finite-width parser limits separate from the BER
    # reader used for canonicalization. Failed parses are ignored by this
    # appliance; they are not additional grounds for rejecting an element.
    reader = Reader(reader.data, appliance=True)
    try:
        root = reader.read(root.start)
    except EncodingError:
        return None
    certificate_ranges, extension_ranges = appliance_ranges(reader, root)
    first_rejection = None

    def inside(offset, ranges):
        return any(start <= offset < end for start, end in ranges)

    # Each entry includes the universal parent tag and the sibling index.
    stack = [(root, None, 0, True)]
    while stack:
        node, parent_tag, index, is_root = stack.pop()
        strict = (inside(node.start, extension_ranges) or
                  not inside(node.start, certificate_ranges))
        rejected = False
        if strict:
            rejected = not node.minimal_length
            if node.cls == 0:
                if node.constructed and node.tag in (3, 4):
                    rejected = True
                elif not node.constructed:
                    v = reader.value(node)
                    if node.tag == 2:
                        rejected |= (not v or len(v) > 1 and (
                            v[0] == 0 and v[1] < 128 or
                            v[0] == 255 and v[1] >= 128))
                    elif node.tag == 1:
                        rejected |= (parent_tag == 16 and index == 0 and v == b'\x00')
        elif node.is_tag(17) and node.constructed:
            try:
                members = appliance_members(reader, node)
                rejected = any(reader.raw(a) >= reader.raw(b)
                               for a, b in zip(members, members[1:]))
            except EncodingError:
                pass
        if rejected:
            if first_rejection is None or node.start < first_rejection:
                first_rejection = node.start
        if node.constructed:
            try:
                parent = node.tag if node.cls == 0 else None
                stack.extend((child, parent, i, False) for i, child
                             in reversed(list(enumerate(appliance_members(reader, node)))))
            except EncodingError:
                pass
        elif not is_root and node.is_tag(4) and node.stop > node.content:
            if (inside(node.content, extension_ranges) and
                    inside(node.stop - 1, extension_ranges)):
                try:
                    stack.append((reader.read(node.content), None, 0, True))
                except EncodingError:
                    pass
    return first_rejection


def minimal_integer(value):
    p = 0
    while p + 1 < len(value) and (
            value[p] == 0 and value[p + 1] < 128 or
            value[p] == 255 and value[p + 1] >= 128):
        p += 1
    return value[p:]


def integer_value(reader, node):
    return int.from_bytes(reader.value(node), 'big', signed=True)


SIGNED_DATA = bytes.fromhex('2a864886f70d010702')
COUNTER_SIGNATURE = bytes.fromhex('2a864886f70d010906')
RSA_PSS = bytes.fromhex('2a864886f70d01010a')
RSA_OAEP = bytes.fromhex('2a864886f70d010107')
SHA1_ALGORITHM = bytes.fromhex('300906052b0e03021a0500')
MGF1_SHA1 = bytes.fromhex('301606092a864886f70d010108300906052b0e03021a0500')
PSPECIFIED_EMPTY = bytes.fromhex('300d06092a864886f70d0101090400')


def label_schema(reader, root):
    """Identify implicit SET OFs and DEFAULTs without guessing from tag alone.

    OCTET STRING and BIT STRING values are bytes, not arbitrary TLV containers.
    In particular, canonicalization preserves extnValue octets and signature
    octets. The appliance's additional parsing of extnValue is a separate
    acceptance rule, not permission to change an OCTET STRING's abstract value.
    """
    root.role = 'content_info'
    stack = [root]
    order = []
    while stack:
        node = stack.pop()
        order.append(node)
        children = reader.children(node)
        role = node.role
        if role == 'content_info' and len(children) >= 2:
            if children[0].is_tag(6) and reader.value(children[0]) == SIGNED_DATA:
                children[1].role = 'signed_data_wrapper'
        elif role == 'signed_data_wrapper' and children:
            children[0].role = 'signed_data'
        elif role == 'signed_data':
            for i, child in enumerate(children):
                if i == 1:
                    child.role = 'algorithms'
                elif i >= 3:
                    if child.is_tag(0, 128):
                        child.role = 'certificates'
                    elif child.is_tag(1, 128):
                        child.role = 'revocations'
                    elif child.is_tag(17):
                        child.role = 'signers'
        elif role == 'certificates':
            for child in children:
                if child.is_tag(16):
                    child.role = 'certificate'
                elif child.is_tag(2, 128):
                    child.role = 'attribute_certificate'
        elif role == 'certificate' and children:
            children[0].role = 'tbs_certificate'
            if len(children) > 1:
                children[1].role = 'algorithm'
        elif role == 'tbs_certificate':
            base = 0
            if children and children[0].is_tag(0, 128):
                version = reader.children(children[0])
                if (len(version) == 1 and version[0].is_tag(2) and
                        integer_value(reader, version[0]) == 0):
                    children[0].omit = True
                base = 1
            if len(children) > base + 1:
                children[base + 1].role = 'algorithm'
            if len(children) > base + 5:
                children[base + 5].role = 'public_key_info'
            for child in children[base + 6:]:
                if child.is_tag(3, 128):
                    child.role = 'extensions_wrapper'
                elif child.is_tag(1, 128) or child.is_tag(2, 128):
                    child.role = 'implicit_bits'
        elif role == 'public_key_info' and children:
            children[0].role = 'algorithm'
        elif role == 'extensions_wrapper' and children:
            children[0].role = 'extensions'
        elif role == 'extensions':
            for child in children:
                child.role = 'extension'
        elif role == 'extension':
            if len(children) >= 3 and children[1].is_tag(1):
                if not any(reader.value(children[1])):
                    children[1].omit = True
        elif role == 'revocations':
            for child in children:
                if child.is_tag(16):
                    child.role = 'crl'
        elif role == 'crl' and children:
            children[0].role = 'tbs_crl'
            if len(children) > 1:
                children[1].role = 'algorithm'
        elif role == 'tbs_crl':
            base = 1 if children and children[0].is_tag(2) else 0
            if len(children) > base:
                children[base].role = 'algorithm'
            for child in children[base + 3:]:
                if child.is_tag(0, 128):
                    child.role = 'extensions_wrapper'
                elif child.is_tag(16):
                    child.role = 'revoked_entries'
        elif role == 'revoked_entries':
            for child in children:
                child.role = 'revoked_entry'
        elif role == 'revoked_entry' and len(children) > 2:
            children[2].role = 'extensions'
        elif role == 'attribute_certificate' and children:
            children[0].role = 'attribute_certificate_info'
            if len(children) > 1:
                children[1].role = 'algorithm'
        elif role == 'attribute_certificate_info':
            if len(children) > 3:
                children[3].role = 'algorithm'
            if len(children) > 7 and children[-1].is_tag(16):
                children[-1].role = 'extensions'
        elif role == 'signers':
            for child in children:
                child.role = 'signer'
        elif role == 'signer':
            if len(children) > 1 and children[1].is_tag(0, 128):
                children[1].role = 'implicit_octets'
            if len(children) > 2:
                children[2].role = 'algorithm'
            pos = 3
            if len(children) > pos and children[pos].is_tag(0, 128):
                children[pos].role = 'attributes'
                pos += 1
            if len(children) > pos:
                children[pos].role = 'algorithm'
            for child in children[pos + 2:]:
                if child.is_tag(1, 128):
                    child.role = 'attributes'
        elif role == 'attributes':
            for child in children:
                child.role = 'attribute'
        elif role == 'attribute' and len(children) > 1:
            if children[0].is_tag(6) and reader.value(children[0]) == COUNTER_SIGNATURE:
                children[1].role = 'signers'
        elif role == 'algorithms':
            for child in children:
                child.role = 'algorithm'
        elif role == 'algorithm' and len(children) > 1 and children[0].is_tag(6):
            oid = reader.value(children[0])
            if oid == RSA_PSS:
                children[1].role = 'pss_parameters'
            elif oid == RSA_OAEP:
                children[1].role = 'oaep_parameters'
        elif role in ('pss_parameters', 'oaep_parameters'):
            for child in children:
                if child.cls == 128 and child.tag in (0, 1):
                    for inner in reader.children(child):
                        inner.role = 'algorithm'
        stack.extend(reversed(children))
    return order


def canonical_time(value, generalized):
    """Normalize BER time offsets, precision and fractional notation for DER."""
    import datetime
    import re
    from decimal import Decimal
    try:
        s = value.decode('ascii')
        if generalized:
            match = re.fullmatch(r'(\d{10})(\d{2})?(\d{2})?([.,]\d+)?(Z|[+-]\d{2}(?:\d{2})?)', s)
            if not match:
                return value
            date, minute, second, fraction, zone = match.groups()
            dt = datetime.datetime.strptime(date, '%Y%m%d%H')
            dt += datetime.timedelta(minutes=int(minute or 0), seconds=int(second or 0))
            fractional_seconds = Decimal('0' + (fraction or '.0').replace(',', '.'))
            fractional_seconds *= 1 if second else 60 if minute else 3600
            whole = int(fractional_seconds)
            dt += datetime.timedelta(seconds=whole)
            remainder = fractional_seconds - whole
            suffix = format(remainder, 'f')[1:].rstrip('0') if remainder else ''
        else:
            match = re.fullmatch(r'(\d{10})(\d{2})?(Z|[+-]\d{4})', s)
            if not match:
                return value
            date, second, zone = match.groups()
            year = int(date[:2])
            year += 1900 if year >= 50 else 2000
            dt = datetime.datetime(year, int(date[2:4]), int(date[4:6]),
                                   int(date[6:8]), int(date[8:10]), int(second or 0))
            suffix = ''
        if zone != 'Z':
            offset = int(zone[1:3]) * 60 + int(zone[3:] or 0)
            dt -= datetime.timedelta(minutes=offset if zone[0] == '+' else -offset)
        if generalized:
            stamp = f'{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}{dt.second:02d}'
        else:
            stamp = dt.strftime('%y%m%d%H%M%S')
        return (stamp + suffix + 'Z').encode('ascii')
    except (ValueError, OverflowError, UnicodeError):
        return value


def canonical_der(reader, root):
    order = label_schema(reader, root)
    string_tags = {3, 4, 12, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30}
    implicit_sets = {'certificates', 'revocations', 'attributes'}
    for node in reversed(order):
        children = reader.children(node)
        constructed = node.constructed
        value = reader.value(node)
        string_tag = node.tag if node.cls == 0 and node.tag in string_tags else None
        if node.role == 'implicit_octets':
            string_tag = 4
        elif node.role == 'implicit_bits':
            string_tag = 3
        if string_tag is not None and constructed:
            # payload is the canonical primitive content, so this also handles
            # nested constructed segments, not just one level of fragments.
            if string_tag == 3:
                fragments = [c.payload for c in children]
                if any(not f for f in fragments):
                    raise EncodingError('empty BIT STRING segment')
                if any(f[0] != 0 for f in fragments[:-1]):
                    raise EncodingError('non-final BIT STRING segment has unused bits')
                value = bytes((fragments[-1][0] if fragments else 0,))
                value += b''.join(f[1:] for f in fragments)
            else:
                value = b''.join(c.payload for c in children)
            constructed = False
        elif constructed:
            if node.role in ('pss_parameters', 'oaep_parameters'):
                defaults = {0: SHA1_ALGORITHM, 1: MGF1_SHA1}
                if node.role == 'pss_parameters':
                    defaults.update({2: b'\x02\x01\x14', 3: b'\x02\x01\x01'})
                else:
                    defaults[2] = PSPECIFIED_EMPTY
                for c in children:
                    if c.cls == 128 and c.payload == defaults.get(c.tag):
                        c.omit = True
            members = [c.der for c in children if not c.omit]
            if node.is_tag(17) or node.role in implicit_sets:
                members.sort()
            value = b''.join(members)
        elif node.cls == 0:
            if node.tag in (2, 10):
                value = minimal_integer(value)
            elif node.tag == 1:
                value = b'\xff' if any(value) else b'\x00'
            elif node.tag in (6, 13):
                # Re-encode each base-128 subidentifier without redundant groups.
                out = bytearray()
                subidentifier = 0
                for b in value:
                    subidentifier = (subidentifier << 7) | (b & 127)
                    if b < 128:
                        groups = [subidentifier & 127]
                        subidentifier >>= 7
                        while subidentifier:
                            groups.append(128 | (subidentifier & 127))
                            subidentifier >>= 7
                        out.extend(reversed(groups))
                value = bytes(out)
            elif node.tag in (23, 24):
                value = canonical_time(value, node.tag == 24)
        if string_tag == 3 and value:
            unused = value[0]
            if unused > 7 or len(value) == 1 and unused:
                raise EncodingError('invalid BIT STRING padding')
            if unused:
                value = value[:-1] + bytes((value[-1] & (255 << unused),))
        node.payload = value
        node.der = tlv(node.cls, constructed, node.tag, value)
    return root.der


def adjudicate(data):
    reader = Reader(data)
    root = reader.read(0)
    rejecting_offset = appliance_rejection(reader, root)
    der = canonical_der(reader, root)
    return {
        'canonical_der': base64.b64encode(der).decode('ascii'),
        'verdict': 'accept' if rejecting_offset is None else 'reject',
        'first_rejecting_element': rejecting_offset,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, type=Path)
    args = parser.parse_args()
    for path in sorted(args.corpus.glob('*.der'), key=lambda p: p.name):
        if path.is_file():
            record = {'object': path.name}
            record.update(adjudicate(path.read_bytes()))
            print(json.dumps(record, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
