#!/usr/bin/env python3
"""Canonicalise BER CMS objects and reproduce the storage appliance's checks.

The BER reader retains source coordinates.  The encoding and acceptance
passes deliberately use different rules: acceptance is not a DER test.
"""

import argparse
import base64
import json
from pathlib import Path
import sys


# BER permits segmentation of these universal string types.  The implicit
# string fields in CMS and certificates are marked by the schema pass below.
STRING_TAGS = frozenset((3, 4, 12, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30))


def der_length(size):
    if size < 128:
        return bytes((size,))
    value = size.to_bytes((size.bit_length() + 7) // 8, "big")
    return bytes((0x80 | len(value),)) + value


def identifier(cls, constructed, tag):
    first = (cls << 6) | (0x20 if constructed else 0)
    if tag < 31:
        return bytes((first | tag,))
    digits = [tag & 127]
    tag >>= 7
    while tag:
        digits.append(0x80 | (tag & 127))
        tag >>= 7
    return bytes((first | 31,)) + bytes(reversed(digits))


def minimal_integer(value):
    pos = 0
    while pos + 1 < len(value):
        if value[pos] == 0 and value[pos + 1] < 128:
            pos += 1
        elif value[pos] == 255 and value[pos + 1] >= 128:
            pos += 1
        else:
            break
    return value[pos:]


class BERError(ValueError):
    pass


class Element:
    __slots__ = ("start", "body", "stop", "end", "cls", "tag", "constructed",
                 "indefinite", "canonical_length", "value", "children",
                 "set_of", "omit", "string_tag")

    def __init__(self):
        self.children = None
        self.set_of = False
        self.omit = False
        self.string_tag = None


class BER:
    def __init__(self, data):
        self.data = data
        self.cache = {}

    def read(self, start):
        cached = self.cache.get(start)
        if cached is not None:
            return cached
        data = self.data
        size = len(data)
        if start >= size:
            raise BERError("missing tag")
        node = Element()
        node.start = start
        octet = data[start]
        node.cls = octet >> 6
        node.constructed = bool(octet & 32)
        node.tag = octet & 31
        pos = start + 1
        if node.tag == 31:
            node.tag = 0
            while True:
                if pos >= size:
                    raise BERError("truncated tag")
                octet = data[pos]
                pos += 1
                node.tag = (node.tag << 7) | (octet & 127)
                if not octet & 128:
                    break
        if pos >= size:
            raise BERError("missing length")
        octet = data[pos]
        pos += 1
        node.indefinite = octet == 128
        node.canonical_length = True
        if node.indefinite:
            node.body = pos
            while True:
                if pos + 2 > size:
                    raise BERError("unterminated indefinite value")
                if data[pos:pos + 2] == b"\x00\x00":
                    break
                child = self.read(pos)
                pos = child.end
            node.stop = pos
            node.end = pos + 2
        else:
            if octet < 128:
                length = octet
            else:
                count = octet & 127
                if pos + count > size:
                    raise BERError("truncated length")
                length = int.from_bytes(data[pos:pos + count], "big")
                # This is also precisely the appliance's length check.
                node.canonical_length = length >= 128 and data[pos] != 0
                pos += count
            node.body = pos
            node.stop = pos + length
            node.end = node.stop
            if node.end > size:
                raise BERError("truncated value")
        node.value = data[node.body:node.stop]
        self.cache[start] = node
        return node

    def members(self, node):
        if node.children is None:
            result = []
            pos = node.body
            while pos < node.stop:
                child = self.read(pos)
                if self.is_eoc(child):
                    break
                result.append(child)
                pos = child.end
            node.children = result
        return node.children

    @staticmethod
    def is_eoc(node):
        return (node.cls == 0 and node.tag == 0 and not node.constructed
                and not node.value)


def is_tag(node, cls, tag, constructed=None):
    return (node.cls == cls and node.tag == tag
            and (constructed is None or node.constructed == constructed))


class Appliance:
    """The appliance has a permissive certificate reader and strict islands.

    SET order is compared using the original wire encodings, not their DER
    replacements.  Indefinite lengths pass both readers.  Neither reader
    performs signature or trust-path verification.
    """

    def __init__(self, ber, root):
        self.ber = ber
        self.certificates = []
        self.payloads = []
        self.rejected = None
        self.wire_cache = {}
        self.discover(root)

    def read(self, start):
        """Read with the appliance's header limits, including in opaque data."""
        if start in self.wire_cache:
            return self.wire_cache[start]
        data = self.ber.data
        pos = start + 1
        try:
            if data[start] & 31 == 31:
                count = 0
                while True:
                    if count == 6:
                        return None
                    octet = data[pos]
                    pos += 1
                    count += 1
                    if not octet & 128:
                        break
            if data[pos] > 128 and data[pos] & 127 > 8:
                return None
            node = self.ber.read(start)
            if node.indefinite:
                pos = node.body
                while pos < node.stop:
                    child = self.read(pos)
                    if child is None:
                        return None
                    pos = child.end
            self.wire_cache[start] = node
            return node
        except (BERError, IndexError):
            self.wire_cache[start] = None
            return None

    def members(self, node):
        result = []
        pos = node.body
        while pos < node.stop:
            child = self.read(pos)
            if child is None or self.ber.is_eoc(child):
                break
            result.append(child)
            pos = child.end
        return result

    def discover(self, root):
        try:
            root = self.read(root.start)
            if root is None or not root.constructed:
                return
            wrapper = self.members(root)[1]
            if not wrapper.constructed:
                return
            signed_data = self.members(wrapper)[0]
            if not signed_data.constructed:
                return
            for field in self.members(signed_data):
                if is_tag(field, 2, 0, True):
                    for cert in self.members(field):
                        if len(self.certificates) == 16:
                            break
                        self.certificates.append((cert.start, cert.end))
            for start, _ in self.certificates:
                cert = self.read(start)
                if not cert.constructed:
                    continue
                tbs = self.members(cert)[0]
                if not tbs.constructed:
                    continue
                for field in self.members(tbs):
                    if not is_tag(field, 2, 3, True):
                        continue
                    extensions = self.members(field)[0]
                    for extension in self.members(extensions):
                        if len(self.payloads) == 128:
                            return
                        parts = self.members(extension)
                        if parts and is_tag(parts[-1], 0, 4):
                            value = parts[-1]
                            self.payloads.append((value.body, value.stop))
        except (BERError, IndexError):
            return

    @staticmethod
    def inside(offset, intervals):
        return any(start <= offset < end for start, end in intervals)

    def reject(self, node):
        if self.rejected is None or node.start < self.rejected:
            self.rejected = node.start

    def check(self, node, parent_tag=-1, index=0):
        strict = (self.inside(node.start, self.payloads)
                  or not self.inside(node.start, self.certificates))
        if not strict:
            if is_tag(node, 0, 17, True):
                previous = None
                for member in self.members(node):
                    encoded = self.ber.data[member.start:member.end]
                    if previous is not None and previous >= encoded:
                        self.reject(node)
                        break
                    previous = encoded
            return
        if not node.canonical_length:
            self.reject(node)
        if node.cls != 0:
            return
        if node.constructed and node.tag in (3, 4):
            self.reject(node)
        if not node.constructed:
            if node.tag == 2 and (not node.value or minimal_integer(node.value) != node.value):
                self.reject(node)
            if (node.tag == 1 and parent_tag == 16 and index == 0
                    and node.value == b"\x00"):
                self.reject(node)

    def walk(self, node, parent_tag=-1, index=0):
        node = self.read(node.start)
        if node is None:
            return
        self.check(node, parent_tag, index)
        if node.constructed:
            tag = node.tag if node.cls == 0 else -1
            for i, child in enumerate(self.members(node)):
                self.walk(child, tag, i)
        elif (is_tag(node, 0, 4) and node.value
              and self.inside(node.body, self.payloads)
              and self.inside(node.stop - 1, self.payloads)):
            try:
                payload = self.read(node.body)
                if payload is not None:
                    self.walk(payload)
            except BERError:
                pass


def mark_schema(ber, root):
    """Identify implicit SET OFs and DEFAULTs in CMS and X.509 structures."""
    def certificate(cert):
        if not is_tag(cert, 0, 16, True):
            return
        fields = ber.members(ber.members(cert)[0])
        for field in fields:
            if is_tag(field, 2, 0, True):
                parts = ber.members(field)
                if (len(parts) == 1 and is_tag(parts[0], 0, 2)
                        and int.from_bytes(parts[0].value, "big", signed=True) == 0):
                    field.omit = True
            elif field.cls == 2 and field.tag in (1, 2):
                # issuerUniqueID and subjectUniqueID are implicit BIT STRINGs.
                field.string_tag = 3
            elif is_tag(field, 2, 3, True):
                for extension in ber.members(ber.members(field)[0]):
                    parts = ber.members(extension)
                    if (len(parts) >= 3 and is_tag(parts[1], 0, 1)
                            and parts[1].value == b"\x00"):
                        parts[1].omit = True

    try:
        outer = ber.members(root)
        # id-signedData (1.2.840.113549.1.7.2)
        if not (is_tag(outer[0], 0, 6)
                and outer[0].value == bytes.fromhex("2a864886f70d010702")):
            return
        signed_data = ber.members(outer[1])[0]
        fields = ber.members(signed_data)
        for field in fields[3:]:
            if is_tag(field, 2, 0, True):
                field.set_of = True
                for cert in ber.members(field):
                    certificate(cert)
            elif is_tag(field, 2, 1, True):
                field.set_of = True
            elif is_tag(field, 0, 17, True):
                for signer in ber.members(field):
                    parts = ber.members(signer)
                    if len(parts) > 1 and is_tag(parts[1], 2, 0):
                        # SignerIdentifier's subjectKeyIdentifier alternative.
                        parts[1].string_tag = 4
                    for part in parts[3:]:
                        if part.cls == 2 and part.tag in (0, 1) and part.constructed:
                            part.set_of = True
    except (BERError, IndexError):
        return


def string_value(ber, node, string_tag):
    if not node.constructed:
        return node.value
    values = [string_value(ber, child, string_tag) for child in ber.members(node)]
    if string_tag != 3:
        return b"".join(values)
    # Each BIT STRING segment carries its own unused-bit count.  BER permits
    # unused bits only in the final segment.
    if not values:
        return b"\x00"
    return values[-1][:1] + b"".join(value[1:] for value in values)


def canonical(ber, node):
    if node.omit:
        return b""
    constructed = node.constructed
    string_tag = node.string_tag
    if node.cls == 0 and node.tag in STRING_TAGS:
        string_tag = node.tag
    if string_tag is not None:
        constructed = False
        value = string_value(ber, node, string_tag)
        if string_tag == 3 and len(value) > 1 and value[0]:
            # DER clears the unused low bits of the last content octet.
            value = value[:-1] + bytes((value[-1] & (255 << value[0]),))
    elif constructed:
        values = [canonical(ber, child) for child in ber.members(node)]
        if node.set_of or is_tag(node, 0, 17):
            values.sort()
        value = b"".join(values)
    else:
        value = node.value
        if node.cls == 0:
            if node.tag in (2, 10):
                value = minimal_integer(value)
            elif node.tag == 1:
                value = b"\xff" if any(value) else b"\x00"
    return identifier(node.cls, constructed, node.tag) + der_length(len(value)) + value


def adjudicate(data):
    # A certificate reader is not limited to the nesting depth of the supplied
    # examples.  At least two input octets are needed for every nested element.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(data) * 2 + 100))
    ber = BER(data)
    root = ber.read(0)
    appliance = Appliance(ber, root)
    appliance.walk(root)
    mark_schema(ber, root)
    encoded = canonical(ber, root)
    return {
        "canonical_der": base64.b64encode(encoded).decode("ascii"),
        "verdict": "accept" if appliance.rejected is None else "reject",
        "first_rejecting_element": appliance.rejected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    args = parser.parse_args()
    for path in sorted((p for p in args.corpus.glob("*.der") if p.is_file()),
                       key=lambda p: p.name):
        result = {"object": path.name}
        result.update(adjudicate(path.read_bytes()))
        print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
