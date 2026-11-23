#!/usr/bin/env python3
"""Canonicalise BER CMS objects and reproduce the signing appliance's reader.

The two operations deliberately use different rules.  DER is schema-aware;
the appliance has a permissive certificate reader, stricter CMS/extension
readers, and compares the *original* encodings when checking certificate SETs.
Only the files selected by --corpus are needed at run time.
"""

import argparse
import base64
import json
from pathlib import Path


def der_length(n):
    if n < 128:
        return bytes((n,))
    octets = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes((128 | len(octets),)) + octets


def der_tag(cls, constructed, number):
    first = cls << 6 | (32 if constructed else 0)
    if number < 31:
        return bytes((first | number,))
    tail = [number & 127]
    number >>= 7
    while number:
        tail.append(128 | (number & 127))
        number >>= 7
    return bytes((first | 31, *reversed(tail)))


def tlv(cls, constructed, number, value):
    return der_tag(cls, constructed, number) + der_length(len(value)) + value


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


class BERError(ValueError):
    pass


class Element:
    __slots__ = ("reader", "start", "cls", "constructed", "number",
                 "length_start", "content", "length", "end", "indefinite",
                 "minimal_length", "_children")

    def __init__(self, reader, start):
        self.reader = reader
        self.start = start
        data = reader.data
        limit = len(data)
        p = start
        if p >= limit:
            raise BERError("Missing tag")
        tag = data[p]
        p += 1
        self.cls = tag >> 6
        self.constructed = bool(tag & 32)
        self.number = tag & 31
        if self.number == 31:
            self.number = 0
            count = 0
            while True:
                if p >= limit or (reader.appliance and count == 6):
                    raise BERError("Incomplete high tag")
                digit = data[p]
                p += 1
                count += 1
                self.number = self.number * 128 + (digit & 127)
                if not digit & 128:
                    break
        self.length_start = p
        if p >= limit:
            raise BERError("Missing length")
        first = data[p]
        p += 1
        self.indefinite = first == 128
        self.minimal_length = True
        self._children = None
        if self.indefinite:
            self.content = p
            # Reader resolves matching EOCs with an explicit stack.
            self.length = None
            self.end = None
        else:
            if first < 128:
                length = first
            else:
                count = first & 127
                if p + count > limit or (reader.appliance and count > 8):
                    raise BERError("Incomplete length")
                length = int.from_bytes(data[p:p + count], "big")
                self.minimal_length = length >= 128 and data[p] != 0
                p += count
            self.content = p
            self.length = length
            self.end = p + length
            if self.end > limit:
                raise BERError("Incomplete contents")

    def is_type(self, number, cls=0):
        return self.cls == cls and self.number == number

    @property
    def value(self):
        return self.reader.data[self.content:self.content + self.length]

    @property
    def raw(self):
        return self.reader.data[self.start:self.end]

    @property
    def eoc(self):
        return self.is_type(0) and not self.constructed and self.length == 0

    def children(self):
        if self._children is None:
            result = []
            if self.constructed:
                p = self.content
                stop = p + self.length
                while p < stop:
                    child = self.reader.read(p)
                    if child.eoc:
                        break
                    result.append(child)
                    p = child.end
            self._children = result
        return self._children


class Reader:
    def __init__(self, data, appliance=False):
        self.data = data
        self.appliance = appliance
        self.cache = {}

    def read(self, offset=0):
        if offset not in self.cache:
            node = Element(self, offset)
            self.cache[offset] = node
            if node.indefinite:
                pending = [node]
                p = node.content
                while pending:
                    if self.data[p:p + 2] == b"\x00\x00":
                        current = pending.pop()
                        current.length = p - current.content
                        current.end = p + 2
                        p += 2
                    else:
                        child = self.cache.get(p)
                        if child is None:
                            child = Element(self, p)
                            self.cache[p] = child
                        if child.indefinite and child.end is None:
                            pending.append(child)
                            p = child.content
                        else:
                            p = child.end
        node = self.cache[offset]
        if node.end is None:
            raise BERError("Unterminated contents")
        return node


def in_ranges(offset, ranges):
    return any(start <= offset < end for start, end in ranges)


class Appliance:
    """Model the appliance's TLV traversal, including its reader boundaries."""

    def __init__(self, reader):
        self.reader = reader
        self.certificates = []
        self.payloads = []
        self.first = None

    @staticmethod
    def children(node):
        # A parse failure stops this sibling scan, but does not discard the
        # elements already visited.  This also matters for opaque octets that
        # the appliance attempts to interpret as an extension value.
        children = []
        if node.constructed:
            p = node.content
            while p < node.content + node.length:
                try:
                    child = node.reader.read(p)
                except BERError:
                    break
                if child.eoc:
                    break
                children.append(child)
                p = child.end
        return children

    def discover(self, root):
        # ContentInfo.content -> SignedData -> certificates [0].  The reader
        # recognises these by position/tag, not by trust or certificate OIDs.
        try:
            signed = self.children(self.children(root)[1])[0]
        except (BERError, IndexError):
            return
        certs = []
        for bag in self.children(signed):
            if bag.is_type(0, 2) and bag.constructed:
                certs.extend(self.children(bag)[:16 - len(certs)])
            if len(certs) >= 16:
                break
        self.certificates = [(c.start, c.end) for c in certs]
        for cert in certs:
            try:
                tbs = self.children(cert)[0]
                for wrapper in self.children(tbs):
                    if not (wrapper.is_type(3, 2) and wrapper.constructed):
                        continue
                    extensions = self.children(wrapper)[0]
                    for extension in self.children(extensions):
                        if len(self.payloads) >= 128:
                            break
                        fields = self.children(extension)
                        if fields and fields[-1].is_type(4):
                            value = fields[-1]
                            self.payloads.append((value.content,
                                                  value.content + value.length))
            except (BERError, IndexError):
                continue

    def reject(self, node):
        if self.first is None or node.start < self.first:
            self.first = node.start

    def check(self, node, parent=None, index=0):
        lax = (in_ranges(node.start, self.certificates)
               and not in_ranges(node.start, self.payloads))
        if lax:
            if node.is_type(17) and node.constructed:
                members = self.children(node)
                if any(a.raw >= b.raw for a, b in zip(members, members[1:])):
                    self.reject(node)
            return
        # Indefinite length is intentionally accepted.  The appliance checks
        # minimality only for definite lengths, rather than requiring DER.
        if not node.minimal_length:
            self.reject(node)
        if node.cls == 0 and node.constructed and node.number in (3, 4):
            self.reject(node)
        if node.cls == 0 and not node.constructed:
            if node.number == 2:
                if not node.value or minimal_integer(node.value) != node.value:
                    self.reject(node)
            elif (node.number == 1 and node.value == b"\x00"
                  and parent is not None and parent.is_type(16) and index == 0):
                self.reject(node)

    def walk(self, root):
        # An explicit stack keeps traversal independent of Python's recursion
        # limit.  Embedded extnValue objects retain their absolute file offsets.
        pending = [(root, None, 0, True)]
        while pending:
            node, parent, index, root_call = pending.pop()
            self.check(node, parent, index)
            if node.constructed:
                try:
                    pending.extend((c, node, i, False)
                                   for i, c in enumerate(self.children(node)))
                except BERError:
                    pass
            if (not root_call and node.is_type(4) and not node.constructed
                    and node.length and in_ranges(node.content, self.payloads)
                    and in_ranges(node.content + node.length - 1, self.payloads)):
                try:
                    pending.append((self.reader.read(node.content), None, 0, True))
                except BERError:
                    pass

    def verdict(self):
        try:
            root = self.reader.read()
            self.discover(root)
            self.walk(root)
        except BERError:
            pass
        return self.first


def string_contents(node, number):
    """Flatten BER string fragments, retaining BIT STRING unused-bit counts."""
    if not node.constructed:
        return node.value
    chunks = []
    pending = [node]
    while pending:
        current = pending.pop()
        if current.constructed:
            pending.extend(reversed(current.children()))
        else:
            chunks.append(current.value)
    if number == 3:
        if not chunks:
            return b"\x00"
        return chunks[-1][:1] + b"".join(c[1:] for c in chunks)
    return b"".join(chunks)


def oid(node):
    if not node.is_type(6):
        return ""
    numbers = []
    number = 0
    for digit in node.value:
        number = number * 128 + (digit & 127)
        if not digit & 128:
            numbers.append(number)
            number = 0
    if not numbers:
        return ""
    first = numbers.pop(0)
    arc = min(first // 40, 2)
    return ".".join(map(str, [arc, first - arc * 40] + numbers))


class Canonicaliser:
    """DER encoder with CMS, X.509 and common algorithm DEFAULT semantics."""

    def __init__(self):
        self.roles = {}
        self.omitted = set()
        self.sets = set()
        self.embedded = {}

    def mark(self, node, role):
        self.roles[node] = role

    @staticmethod
    def integer_is(node, value):
        return bool(node.value) and int.from_bytes(node.value, "big", signed=True) == value

    def certificate(self, node):
        fields = node.children()
        if not fields:
            return
        tbs = fields[0]
        parts = tbs.children()
        if parts and parts[0].is_type(0, 2):
            version = parts[0].children()
            if (len(version) == 1 and version[0].is_type(2)
                    and self.integer_is(version[0], 0)):
                self.omitted.add(parts[0])
        for part in parts:
            if part.cls == 2 and part.number in (1, 2):
                self.mark(part, "implicit_bit_string")
            if part.is_type(3, 2) and part.constructed:
                for extseq in part.children():
                    self.extensions(extseq)

    def extensions(self, sequence):
        for extension in sequence.children():
            fields = extension.children()
            if len(fields) < 2:
                continue
            ext_oid = oid(fields[0])
            for field in fields[1:-1]:
                if field.is_type(1) and not any(field.value):
                    self.omitted.add(field)
            value = fields[-1]
            if not value.is_type(4):
                continue
            encoded = string_contents(value, 4)
            try:
                sub = Reader(encoded).read()
                if sub.end != len(encoded):
                    continue
                # An extension's octets encode its ASN.1 value.  Other OCTET
                # STRINGs (digests, signatures, identifiers) remain opaque.
                self.embedded[value] = sub
                if ext_oid == "2.5.29.19":  # BasicConstraints.cA DEFAULT FALSE
                    children = sub.children()
                    if children and children[0].is_type(1) and not any(children[0].value):
                        self.omitted.add(children[0])
                elif ext_oid == "2.5.29.30":  # NameConstraints / GeneralSubtree
                    for subtrees in sub.children():
                        for subtree in subtrees.children():
                            for bound in subtree.children()[1:]:
                                if bound.is_type(0, 2) and self.integer_is(bound, 0):
                                    self.omitted.add(bound)
                                elif bound.cls == 2 and bound.number in (0, 1):
                                    self.mark(bound, "implicit_integer")
                elif ext_oid == "2.5.29.28":  # IssuingDistributionPoint
                    for field in sub.children():
                        if field.cls == 2 and field.number in (1, 2, 4, 5):
                            if not any(field.value):
                                self.omitted.add(field)
                            else:
                                self.mark(field, "implicit_boolean")
                        elif field.is_type(3, 2):
                            self.mark(field, "named_bits")
                elif ext_oid == "2.5.29.15":  # KeyUsage is a named bit list.
                    self.mark(sub, "named_bits")
            except (BERError, IndexError):
                continue

    def crl(self, node):
        fields = node.children()
        if not fields:
            return
        parts = fields[0].children()
        for part in parts:
            if part.is_type(0, 2):
                for extension_sequence in part.children():
                    self.extensions(extension_sequence)
        # After signature, issuer, thisUpdate and optional nextUpdate comes
        # the optional SEQUENCE OF revoked certificates.
        i = 1 if parts and parts[0].is_type(2) else 0
        i += 3
        if i < len(parts) and parts[i].cls == 0 and parts[i].number in (23, 24):
            i += 1
        if i < len(parts) and parts[i].is_type(16):
            for entry in parts[i].children():
                entry_fields = entry.children()
                if len(entry_fields) > 2:
                    self.extensions(entry_fields[2])

    def attributes(self, node):
        self.sets.add(node)
        for attribute in node.children():
            fields = attribute.children()
            if len(fields) != 2:
                continue
            attribute_oid = oid(fields[0])
            if attribute_oid == "1.2.840.113549.1.9.6":  # counterSignature
                for signer in fields[1].children():
                    self.signer(signer)
            elif attribute_oid == "1.2.840.113549.1.9.16.2.14":
                for content_info in fields[1].children():
                    self.cms(content_info)

    def signer(self, node):
        fields = node.children()
        if len(fields) < 3:
            return
        if fields[1].is_type(0, 2):
            self.mark(fields[1], "implicit_octet_string")
        for field in fields[3:]:
            if field.cls == 2 and field.number in (0, 1):
                self.attributes(field)

    def cms(self, root):
        fields = root.children()
        if len(fields) < 2 or not fields[1].children():
            return
        signed = fields[1].children()[0]
        if oid(fields[0]) != "1.2.840.113549.1.7.2":
            return
        parts = signed.children()
        for part in parts[3:]:
            if part.is_type(0, 2):
                self.sets.add(part)
                for certificate in part.children():
                    if certificate.is_type(16):
                        self.certificate(certificate)
            elif part.is_type(1, 2):
                self.sets.add(part)
                for crl in part.children():
                    if crl.is_type(16):
                        self.crl(crl)
            elif part.is_type(17):
                for signer in part.children():
                    self.signer(signer)

    def algorithm_defaults(self, root):
        # These defaults are defined by the algorithm's parameter type, not
        # by the tag/position of an arbitrary INTEGER or BOOLEAN.
        sha1 = bytes.fromhex("300906052b0e03021a0500")
        mgf1_sha1 = bytes.fromhex("301606092a864886f70d010108") + sha1
        psource_empty = bytes.fromhex("300d06092a864886f70d0101090400")
        hmac_sha1 = bytes.fromhex("300c06082a864886f70d02070500")
        pending = [root, *self.embedded.values()]
        while pending:
            node = pending.pop()
            children = node.children()
            pending.extend(children)
            if not node.is_type(16) or len(children) != 2 or not children[1].is_type(16):
                continue
            algorithm = oid(children[0])
            parameters = children[1].children()
            if algorithm in ("1.2.840.113549.1.1.10", "1.2.840.113549.1.1.7"):
                pss = algorithm.endswith(".10")
                for parameter in parameters:
                    if parameter.cls != 2 or len(parameter.children()) != 1:
                        continue
                    inner = parameter.children()[0]
                    number = parameter.number
                    default = None
                    if number == 0:
                        default = sha1
                    elif number == 1:
                        default = mgf1_sha1
                    elif number == 2 and not pss:
                        default = psource_empty
                    if default is not None and self.encode(inner) == default:
                        self.omitted.add(parameter)
                    elif (pss and number in (2, 3) and inner.is_type(2)
                          and self.integer_is(inner, 20 if number == 2 else 1)):
                        self.omitted.add(parameter)
            elif algorithm == "1.2.840.113549.1.5.12" and parameters:
                if self.encode(parameters[-1]) == hmac_sha1:
                    self.omitted.add(parameters[-1])

    def encode(self, root):
        encoded = {}
        pending = [(root, False)]
        string_tags = (3, 4, 7, 12, 18, 19, 20, 21, 22, 25, 26, 27, 28, 30)
        while pending:
            node, ready = pending.pop()
            if node in self.omitted:
                encoded[node] = b""
                continue
            role = self.roles.get(node)
            is_string = ((node.cls == 0 and node.number in string_tags)
                         or role in ("implicit_bit_string", "implicit_octet_string", "named_bits"))
            embedded = self.embedded.get(node)
            if not ready:
                if is_string and embedded is not None:
                    pending.extend(((node, True), (embedded, False)))
                    continue
                if node.constructed and not is_string:
                    pending.append((node, True))
                    pending.extend((child, False) for child in node.children())
                    continue
            constructed = node.constructed
            if is_string:
                number = node.number
                if role in ("implicit_bit_string", "named_bits"):
                    number = 3
                elif role == "implicit_octet_string":
                    number = 4
                value = encoded[embedded] if embedded is not None else string_contents(node, number)
                if number == 3 and value:
                    unused = value[0]
                    bits = bytearray(value[1:])
                    if bits and 0 < unused < 8:
                        bits[-1] &= (255 << unused) & 255
                    if role == "named_bits":
                        while bits and bits[-1] == 0:
                            bits.pop()
                        unused = ((bits[-1] & -bits[-1]).bit_length() - 1) if bits else 0
                    value = bytes((unused,)) + bytes(bits)
                constructed = False
            elif constructed:
                members = [encoded[child] for child in node.children()]
                if node.is_type(17) or node in self.sets:
                    members.sort()
                value = b"".join(members)
            else:
                value = node.value
                if node.is_type(2) or node.is_type(10) or role == "implicit_integer":
                    value = minimal_integer(value)
                elif node.is_type(1) or role == "implicit_boolean":
                    value = b"\xff" if any(value) else b"\x00"
            encoded[node] = tlv(node.cls, constructed, node.number, value)
        return encoded[root]

    def canonicalise(self, root):
        self.cms(root)
        self.algorithm_defaults(root)
        return self.encode(root)


def adjudicate(data):
    reader = Reader(data)
    offset = Appliance(Reader(data, appliance=True)).verdict()
    canonical = Canonicaliser().canonicalise(reader.read())
    return canonical, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    args = parser.parse_args()
    for path in sorted(args.corpus.glob("*.der"), key=lambda p: p.name):
        if not path.is_file():
            continue
        canonical, offset = adjudicate(path.read_bytes())
        print(json.dumps({"object": path.name,
                          "canonical_der": base64.b64encode(canonical).decode("ascii"),
                          "verdict": "accept" if offset is None else "reject",
                          "first_rejecting_element": offset}, separators=(",", ":")))


if __name__ == "__main__":
    main()
