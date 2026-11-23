#!/usr/bin/env python3
"""Build-time corpus generator for the CMS trust-store acceptance task.

This script is executed inside the *builder* stage of the environment image.
It is never copied into the agent image.

Every object is produced twice over:

  * as a tree of ASN.1 nodes carrying, per node, an "encoding style" that
    decides whether the node is written canonically or in one of the
    deliberately non-canonical forms the task is about, and
  * as the BER/DER byte string that the tree encodes to.

The reference verdict/offset for an object is computed by re-parsing the
*encoded bytes* with an independent little model of the appliance, never by
trusting the recipe that produced them.  The recipe's own expectation is
cross-checked against that model at build time, and the compiled `accepts`
binary is cross-checked against both, so the three cannot drift apart.

Usage:
    generate_corpus.py --out DIR [--split visible|heldout|all]
                       [--accepts /path/to/accepts] [--seed N]
"""

import argparse
import base64
import json
import os
import random
import subprocess
import sys

SEED = 0x53454C46524553  # fixed: "SELFRES"

UNIV = 0x00
CTX = 0x80

# --------------------------------------------------------------------- nodes


class N(object):
    """A single TLV node.

    kids is None for a primitive, a (possibly empty) list for a constructed.
    For a constructed OCTET/BIT STRING that must re-encode as primitive, the
    primitive content is kept in `prim`.
    """

    __slots__ = ("cls", "num", "cons", "val", "kids", "setof", "dflt",
                 "prim", "lstyle", "off", "cval")

    def __init__(self, cls, num, cons=False, val=None, kids=None, setof=False,
                 dflt=None, prim=None, lstyle="min", cval=None):
        self.cls = cls
        self.num = num
        self.cons = cons
        self.val = val
        self.kids = kids
        self.setof = setof
        self.dflt = dflt
        self.prim = prim
        self.lstyle = lstyle
        self.off = None
        self.cval = cval


def tagb(cls, cons, num):
    b = cls | (0x20 if cons else 0)
    if num < 31:
        return bytes([b | num])
    out = [b | 0x1F]
    d = []
    while num:
        d.append(num & 0x7F)
        num >>= 7
    d.reverse()
    for x in d[:-1]:
        out.append(x | 0x80)
    out.append(d[-1])
    return bytes(out)


def lenb(style, n, cons):
    if style == "indef":
        assert cons, "indefinite length on a primitive"
        return b"\x80"
    if style == "pad":
        k = max(1, (n.bit_length() + 7) // 8) + 1
        return bytes([0x80 | k]) + n.to_bytes(k, "big")
    if n < 0x80:
        return bytes([n])
    d = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(d)]) + d


DEFERRED = []


def fix_deferred():
    """Give nodes that live inside a primitive's value their real offsets."""
    for inner, holder in DEFERRED:
        if holder.prim is not None:
            continue
        hdr = tagb(holder.cls, holder.cons, holder.num)
        cbase = holder.off + len(hdr) + len(lenb(holder.lstyle, len(holder.val), holder.cons))
        shift(inner, cbase)
    del DEFERRED[:]


def shift(n, delta):
    n.off += delta
    if n.kids:
        for k in n.kids:
            shift(k, delta)


def enc(n, base):
    """Serialise `n`, recording each node's absolute tag offset."""
    rel = []
    if n.kids is None:
        body = n.val if n.val is not None else b""
        cls, num, cons = n.cls, n.num, n.cons
    else:
        sub = bytearray()
        for k in n.kids:
            rel.append(len(sub))
            sub += enc(k, 0)
        body = bytes(sub)
        cls, num, cons = n.cls, n.num, n.cons
    hdr = tagb(cls, cons, num) + lenb(n.lstyle, len(body), cons)
    n.off = base
    if n.kids is not None:
        for k, r in zip(n.kids, rel):
            shift(k, base + len(hdr) + r)
    if n.lstyle == "indef":
        return hdr + body + b"\x00\x00"
    return hdr + body


def int_min(v):
    """Strip redundant leading octets from a two's-complement value."""
    i = 0
    while len(v) - i > 1:
        if v[i] == 0x00 and v[i + 1] < 0x80:
            i += 1
        elif v[i] == 0xFF and v[i + 1] >= 0x80:
            i += 1
        else:
            break
    return v[i:]


def canon(n):
    """DER re-encoding of the abstract value `n` carries, or None if omitted."""
    if n.cval is not None:
        content = n.cval
        cls, num, cons = n.cls, n.num, False
    elif n.prim is not None:
        content = n.prim
        cls, num, cons = n.cls, n.num, False
    elif n.kids is None:
        content = n.val if n.val is not None else b""
        if n.cls == UNIV and n.num == 2:
            content = int_min(content)
        cls, num, cons = n.cls, n.num, n.cons
    else:
        parts = []
        for k in n.kids:
            e = canon(k)
            if e is not None:
                parts.append(e)
        if n.setof:
            parts.sort()
        content = b"".join(parts)
        cls, num, cons = n.cls, n.num, n.cons
    if n.dflt is not None and content == n.dflt:
        return None
    return tagb(cls, cons, num) + lenb("min", len(content), cons) + content


# ---------------------------------------------------------------- ASN.1 sugar


def seq(*kids):
    return N(UNIV, 16, cons=True, kids=list(kids))


def st(*kids):
    k = list(kids)
    k.sort(key=canon)
    return N(UNIV, 17, cons=True, kids=k, setof=True)


def ctyset(num, kids):
    k = list(kids)
    k.sort(key=canon)
    return N(CTX, num, cons=True, kids=k, setof=True)


def ctx(num, kid):
    return N(CTX, num, cons=True, kids=[kid])


def oid(s):
    return N(UNIV, 6, val=oid_body(s))


def oid_body(s):
    p = [int(x) for x in s.split(".")]
    out = bytearray([p[0] * 40 + p[1]])
    for v in p[2:]:
        if v < 0x80:
            out.append(v)
            continue
        stk = []
        while v:
            stk.append(v & 0x7F)
            v >>= 7
        stk.reverse()
        for x in stk[:-1]:
            out.append(x | 0x80)
        out.append(stk[-1])
    return bytes(out)


def boolean(b):
    return N(UNIV, 1, val=b"\xff" if b else b"\x00")


def null():
    return N(UNIV, 5, val=b"")


def octet(b):
    return N(UNIV, 4, val=b)


def bitstr(b, unused=0):
    return N(UNIV, 3, val=bytes([unused]) + b)


def printable(s):
    return N(UNIV, 19, val=s.encode())


def utf8(s):
    return N(UNIV, 12, val=s.encode())


def utctime(s):
    return N(UNIV, 23, val=s.encode())


def mkint(v, pad=False):
    n = 1
    while True:
        try:
            b = v.to_bytes(n, "big", signed=True)
            break
        except OverflowError:
            n += 1
    if pad:
        b = (b"\x00" if b[0] < 0x80 else b"\xFF") + b
    return N(UNIV, 2, val=b)


def style(node, mode):
    if mode == "pad":
        node.lstyle = "pad"
    elif mode == "indef":
        node.lstyle = "indef"


def as_constructed(node, chunk=9):
    """Rewrite a primitive OCTET/BIT STRING as a constructed one."""
    raw = node.val
    if node.num == 3:
        unused, body = raw[0], raw[1:]
        parts = [body[i:i + chunk] for i in range(0, len(body), chunk)] or [b""]
        kids = [bitstr(p, 0) for p in parts[:-1]] + [bitstr(parts[-1], unused)]
    else:
        parts = [raw[i:i + chunk] for i in range(0, len(raw), chunk)] or [b""]
        kids = [octet(p) for p in parts]
    node.cons = True
    node.prim = raw
    node.val = None
    node.kids = kids


# ---------------------------------------------------------------- identifiers

OID_SIGNED_DATA = "1.2.840.113549.1.7.2"
OID_DATA = "1.2.840.113549.1.7.1"
OID_SHA256 = "2.16.840.1.101.3.4.2.1"
OID_SHA384 = "2.16.840.1.101.3.4.2.2"
OID_EC_PUBLIC_KEY = "1.2.840.10045.2.1"
OID_PRIME256V1 = "1.2.840.10045.3.1.7"
OID_ECDSA_SHA256 = "1.2.840.10045.4.3.2"
OID_CN = "2.5.4.3"
OID_C = "2.5.4.6"
OID_L = "2.5.4.7"
OID_O = "2.5.4.10"
OID_OU = "2.5.4.11"
OID_BC = "2.5.29.19"
OID_KU = "2.5.29.15"
OID_SKI = "2.5.29.14"
OID_AKI = "2.5.29.35"
OID_EKU = "2.5.29.37"
OID_CUSTOM = "1.3.6.1.4.1.41234.9.7"
OID_AT_CONTENT_TYPE = "1.2.840.113549.1.9.3"
OID_AT_MESSAGE_DIGEST = "1.2.840.113549.1.9.4"
OID_AT_SIGNING_TIME = "1.2.840.113549.1.9.5"
OID_EKU_TLS_SERVER = "1.3.6.1.5.5.7.3.1"
OID_EKU_EMAIL = "1.3.6.1.5.5.7.3.4"


def _nm(common, org, unit, city):
    return [
        [(OID_C, printable("US"))],
        [(OID_O, printable(org)), (OID_OU, printable(unit))],
        [(OID_CN, printable(common)), (OID_L, printable(city))],
    ]


def _spec(serial, ca, pathlen, common, org, unit, city, kbyte, keyid):
    return dict(serial=serial, ca=ca, pathlen=pathlen,
                issuer=_nm("Mercy General Root CA", "Mercy General Hospital",
                           "Clinical Engineering", "Springfield"),
                subject=_nm(common, org, unit, city),
                public=b"\x04" + bytes([(i * 37 + kbyte) & 0xFF for i in range(64)]),
                keyid=bytes([keyid]) * 20)


CERT_SPECS = [
    _spec(0x01A1B2C3, True, 1, "Mercy General Root CA", "Mercy General Hospital",
          "Clinical Engineering", "Springfield", 1, 0x11),
    _spec(0x02D4E5F6, True, 0, "Mercy General Issuing CA", "Mercy General Hospital",
          "Clinical Engineering", "Springfield", 11, 0x22),
    _spec(0x03010203, False, None, "Lab Signing Appliance 01", "Mercy General Hospital",
          "Clinical Engineering", "Springfield", 23, 0x33),
    _spec(0x04A0B0C0, False, None, "Lab Signing Appliance 02", "Mercy General Hospital",
          "Pathology", "Springfield", 37, 0x44),
    _spec(0x05556677, False, None, "Records Station 3", "Mercy General Hospital",
          "Health Information Management", "Riverton", 51, 0x55),
    _spec(0x06CCDDEE, False, None, "Vendor Relay", "Northwind Diagnostics",
          "Integration Services", "Fairview", 67, 0x66),
]


# ------------------------------------------------------------------ builders


def algid(o, parm=None):
    if parm is None:
        return seq(oid(o))
    return seq(oid(o), parm)


def rdn_set(atvs):
    k = [seq(oid(t), v) for (t, v) in atvs]
    k.sort(key=canon)
    return N(UNIV, 17, cons=True, kids=k, setof=True)


def build_name(rdns):
    return seq(*[rdn_set(r) for r in rdns])


def build_extensions(spec, quirks, reg):
    q = quirks.get
    exts = []

    kids = []
    if q("ext.inner.bc.ca") == "default":
        b = boolean(False)
        b.dflt = b"\x00"
        kids.append(b)
        reg["ext.inner.bc.ca"] = (b, "LIB")
    elif spec["ca"]:
        kids.append(boolean(True))
    if spec["pathlen"] is not None:
        kids.append(mkint(spec["pathlen"], pad=(q("ext.inner.bc.pathlen") == "pad")))
        if q("ext.inner.bc.pathlen"):
            reg["ext.inner.bc.pathlen"] = (kids[-1], "LIB")
    bc = seq(*kids)
    if q("ext.inner.bc.seq"):
        style(bc, q("ext.inner.bc.seq"))
        reg["ext.inner.bc.seq"] = (bc, "LIB")
    critq = (q("cert.critical") == "default")
    valcons = (q("cert.extvalue") == "cons")
    exts.append(ext_node(OID_BC, bc, critical=not critq, critical_quirk=critq,
                         reg=reg, valcons=valcons, key="cert.extvalue"))

    ku = bitstr(b"\xA0", 5)
    if q("ext.inner.ku") == "cons":
        as_constructed(ku)
        reg["ext.inner.ku"] = (ku, "LIB")
    exts.append(ext_node(OID_KU, ku, critical=True))

    ski = octet(spec["keyid"])
    if q("ext.inner.ski") == "cons":
        as_constructed(ski)
        reg["ext.inner.ski"] = (ski, "LIB")
    exts.append(ext_node(OID_SKI, ski))

    kid = N(CTX, 0, val=spec["keyid"])
    if q("ext.inner.aki") == "pad":
        style(kid, "pad")
        reg["ext.inner.aki"] = (kid, "LIB")
    exts.append(ext_node(OID_AKI, seq(kid)))

    eku = seq(oid(OID_EKU_TLS_SERVER), oid(OID_EKU_EMAIL))
    if q("ext.inner.eku"):
        style(eku, q("ext.inner.eku"))
        reg["ext.inner.eku"] = (eku, "LIB")
    exts.append(ext_node(OID_EKU, eku))

    cset = st(mkint(3), mkint(1), mkint(2))
    if q("ext.inner.custom.set") == "unsorted":
        cset.kids.reverse()
        reg["ext.inner.custom.set"] = (cset, "LIB")
    cs = seq(cset)
    if q("ext.inner.custom.seq"):
        style(cs, q("ext.inner.custom.seq"))
        reg["ext.inner.custom.seq"] = (cs, "LIB")
    exts.append(ext_node(OID_CUSTOM, cs))

    return seq(*exts)


def ext_node(o, inner, critical=False, critical_quirk=False, reg=None,
             valcons=False, key=None):
    kids = [oid(o)]
    if critical:
        kids.append(boolean(True))
    elif critical_quirk:
        b = boolean(False)
        b.dflt = b"\x00"
        kids.append(b)
        if reg is not None:
            reg["cert.critical"] = (b, "CERT")
    raw = enc(inner, 0)
    ev = octet(raw)
    ev.cval = canon(inner)
    DEFERRED.append((inner, ev))
    if valcons:
        as_constructed(ev)
        reg[key] = (ev, "CERT")
    kids.append(ev)
    return seq(*kids)


def build_cert(spec, quirks, reg):
    q = quirks.get

    serial = mkint(spec["serial"], pad=(q("cert.serial") == "pad"))
    if q("cert.serial"):
        reg["cert.serial"] = (serial, "CERT")

    sigalg = algid(OID_ECDSA_SHA256)
    if q("cert.sigalg"):
        style(sigalg, q("cert.sigalg"))
        reg["cert.sigalg"] = (sigalg, "CERT")

    issuer = build_name(spec["issuer"])
    if q("cert.issuer") is not None:
        sub = issuer.kids[q("cert.issuer")]
        sub.kids.reverse()
        reg["cert.issuer"] = (sub, "CERT")
    if q("cert.issuername"):
        style(issuer, q("cert.issuername"))
        reg["cert.issuername"] = (issuer, "CERT")

    subject = build_name(spec["subject"])
    if q("cert.subject") is not None:
        sub = subject.kids[q("cert.subject")]
        sub.kids.reverse()
        reg["cert.subject"] = (sub, "CERT")

    validity = seq(utctime("230101000000Z"), utctime("330101000000Z"))
    if q("cert.validity"):
        style(validity, q("cert.validity"))
        reg["cert.validity"] = (validity, "CERT")

    spki = seq(algid(OID_EC_PUBLIC_KEY, oid(OID_PRIME256V1)), bitstr(spec["public"], 0))
    if q("cert.spki") == "cons":
        as_constructed(spki.kids[1])
        reg["cert.spki"] = (spki.kids[1], "CERT")

    extwrap = ctx(3, build_extensions(spec, quirks, reg))
    if q("cert.extswrap"):
        style(extwrap, q("cert.extswrap"))
        reg["cert.extswrap"] = (extwrap, "CERT")

    version = ctx(0, mkint(2))
    tbs = seq(version, serial, sigalg, issuer, validity, subject, spki, extwrap)
    if q("cert.tbs"):
        style(tbs, q("cert.tbs"))
        reg["cert.tbs"] = (tbs, "CERT")

    sv = bitstr(sig_of(spec["serial"]), 0)
    if q("cert.sigvalue") == "cons":
        as_constructed(sv)
        reg["cert.sigvalue"] = (sv, "CERT")
    elif q("cert.sigvalue"):
        style(sv, q("cert.sigvalue"))
        reg["cert.sigvalue"] = (sv, "CERT")

    return seq(tbs, sigalg2(spec), sv)


def sigalg2(spec):
    return algid(OID_ECDSA_SHA256)


def sig_of(seed):
    return bytes([(i * 29 + seed) & 0xFF for i in range(71)])


# --------------------------------------------------------------- the envelope


def build_signers(spec, quirks, reg):
    q = quirks.get

    sid_name = build_name(spec["issuer"])
    if q("cms.si.sid.name") is not None:
        sub = sid_name.kids[q("cms.si.sid.name")]
        sub.kids.reverse()
        reg["cms.si.sid.name"] = (sub, "LIB")
    if q("cms.si.sid.namepad"):
        style(sid_name, q("cms.si.sid.namepad"))
        reg["cms.si.sid.namepad"] = (sid_name, "LIB")

    serial = mkint(spec["serial"], pad=(q("cms.si.sid.serial") == "pad"))
    if q("cms.si.sid.serial"):
        reg["cms.si.sid.serial"] = (serial, "LIB")

    attrs = []
    a1 = seq(oid(OID_AT_CONTENT_TYPE), st(oid(OID_DATA)))
    attrs.append(a1)
    digest = octet(sig_of(spec["serial"] + 7)[:32])
    mdset = st(digest)
    if q("cms.si.attr.md.value") == "cons":
        as_constructed(digest)
        reg["cms.si.attr.md.value"] = (digest, "LIB")
    attrs.append(seq(oid(OID_AT_MESSAGE_DIGEST), mdset))
    attrs.append(seq(oid(OID_AT_SIGNING_TIME), st(utctime("230101000000Z"))))
    sa = ctyset(0, attrs)
    if q("cms.si.attrs") == "unsorted":
        sa.kids.reverse()
        reg["cms.si.attrs"] = (sa, "LIB")
    elif q("cms.si.attrs"):
        style(sa, q("cms.si.attrs"))
        reg["cms.si.attrs"] = (sa, "LIB")

    sigalg = algid(OID_ECDSA_SHA256)
    if q("cms.si.sigalg"):
        style(sigalg, q("cms.si.sigalg"))
        reg["cms.si.sigalg"] = (sigalg, "LIB")

    sig = octet(sig_of(spec["serial"] + 101))
    if q("cms.si.signature") == "cons":
        as_constructed(sig)
        reg["cms.si.signature"] = (sig, "LIB")
    elif q("cms.si.signature"):
        style(sig, q("cms.si.signature"))
        reg["cms.si.signature"] = (sig, "LIB")

    return seq(mkint(1), seq(sid_name, serial), algid(OID_SHA256), sa, sigalg, sig)


def build_object(spec, helper_spec, quirks, reg):
    q = quirks.get

    signer = build_signers(spec, quirks, reg)
    infos = [signer]
    if q("cms.signerinfos") == "unsorted":
        infos.append(build_signers(helper_spec, {}, {}))
    sinfos = st(*infos)
    if q("cms.signerinfos") == "unsorted":
        reg["cms.signerinfos"] = (sinfos, "LIB")
    elif q("cms.signerinfos"):
        style(sinfos, q("cms.signerinfos"))
        reg["cms.signerinfos"] = (sinfos, "LIB")

    certs = [build_cert(spec, quirks, reg), build_cert(helper_spec, {}, {})]
    cset = ctyset(0, certs)
    if q("cms.certs") == "unsorted":
        cset.kids.reverse()
        reg["cms.certs"] = (cset, "LIB")
    elif q("cms.certs"):
        style(cset, q("cms.certs"))
        reg["cms.certs"] = (cset, "LIB")

    dalgs = st(algid(OID_SHA256), algid(OID_SHA384))
    if q("cms.digests") == "unsorted":
        dalgs.kids.reverse()
        reg["cms.digests"] = (dalgs, "LIB")
    elif q("cms.digests"):
        style(dalgs, q("cms.digests"))
        reg["cms.digests"] = (dalgs, "LIB")

    ver = mkint(1, pad=(q("cms.sdversion") == "pad"))
    if q("cms.sdversion"):
        reg["cms.sdversion"] = (ver, "LIB")

    sd = seq(ver, dalgs, seq(oid(OID_DATA)), cset, sinfos)
    if q("cms.sd"):
        style(sd, q("cms.sd"))
        reg["cms.sd"] = (sd, "LIB")

    content = ctx(0, sd)
    if q("cms.content"):
        style(content, q("cms.content"))
        reg["cms.content"] = (content, "LIB")

    root = seq(oid(OID_SIGNED_DATA), content)
    return root


# ------------------------------------------ the appliance, re-implemented once
#
# Everything below reads *bytes*, never the recipe.  It exists so the
# reference verdict/offset recorded for an object is derived the same way the
# shipped binary derives it, from the object alone.


class El(object):
    __slots__ = ("off", "val_off", "val_len", "end", "cls", "cons", "num",
                 "indef", "len_min", "kids")


def parse_el(d, p, n):
    off = p
    b = d[p]
    el = El()
    el.cls = b & 0xC0
    el.cons = bool(b & 0x20)
    el.num = b & 0x1F
    p += 1
    if el.num == 0x1F:
        el.num = 0
        while True:
            x = d[p]
            p += 1
            el.num = (el.num << 7) | (x & 0x7F)
            if not (x & 0x80):
                break
    l = d[p]
    p += 1
    el.len_min = True
    el.indef = False
    if l == 0x80:
        el.indef = True
        vlen = -1
    elif l < 0x80:
        vlen = l
    else:
        k = l & 0x7F
        raw = d[p:p + k]
        p += k
        vlen = int.from_bytes(raw, "big")
        el.len_min = (vlen >= 0x80) and (raw[0] != 0)
    val_off = p
    kids = []
    if el.indef:
        while True:
            if d[p] == 0 and d[p + 1] == 0:
                p += 2
                break
            c = parse_el(d, p, n)
            kids.append(c)
            p = c.end
        el.val_len = p - 2 - val_off
    else:
        e = p + vlen
        if el.cons:
            q = p
            while q < e:
                c = parse_el(d, q, n)
                kids.append(c)
                q = c.end
            if q != e:
                raise ValueError("child overruns its parent")
        p = e
        el.val_len = vlen
    el.off = off
    el.val_off = val_off
    el.end = p
    el.kids = kids
    return el


def locate(d):
    """Byte ranges of the embedded Certificates and of every extnValue body."""
    root = parse_el(d, 0, len(d))
    sd = root.kids[1].kids[0]
    certs = []
    for k in sd.kids:
        if k.cls == CTX and k.num == 0 and k.cons:
            certs = [(c.off, c.end) for c in k.kids]
    pattr = []
    for a, _b in certs:
        cert = parse_el(d, a, len(d))
        for k in cert.kids[0].kids:
            if k.cls == CTX and k.num == 3 and k.cons:
                for e in k.kids[0].kids:
                    ev = e.kids[-1]
                    pattr.append((ev.val_off, ev.val_off + ev.val_len))
    return root, certs, pattr


def raw_sorted(d, el):
    if len(el.kids) < 2:
        return True
    prev = d[el.kids[0].off:el.kids[0].end]
    for k in el.kids[1:]:
        cur = d[k.off:k.end]
        if cur <= prev:
            return False
        prev = cur
    return True


def int_min_bytes(d, el):
    v = d[el.val_off:el.val_off + el.val_len]
    if not v:
        return False
    if len(v) >= 2:
        if v[0] == 0x00 and v[1] < 0x80:
            return False
        if v[0] == 0xFF and v[1] >= 0x80:
            return False
    return True


def model(d):
    """(verdict, first rejecting tag offset or None) for the appliance."""
    root, certs, pattr = locate(d)
    pset = set(pattr)
    bad = [None]

    def note(off):
        if bad[0] is None or off < bad[0]:
            bad[0] = off

    def owner(off):
        for a, b in pattr:
            if a <= off < b:
                return "LIB"
        for a, b in certs:
            if a <= off < b:
                return "CERT"
        return "LIB"

    def check(k, parent, i):
        if owner(k.off) == "CERT":
            if k.cls == UNIV and k.cons and k.num == 17 and not raw_sorted(d, k):
                note(k.off)
            return
        if not k.len_min:
            note(k.off)
        if k.cls == UNIV and k.cons and k.num in (3, 4):
            note(k.off)
        if k.cls == UNIV and k.num == 2 and not k.cons and not int_min_bytes(d, k):
            note(k.off)
        ptag = parent.num if (parent is not None and parent.cls == UNIV) else None
        if (k.cls == UNIV and k.num == 1 and not k.cons
                and d[k.val_off:k.val_off + k.val_len] == b"\x00"
                and ptag == 16 and i == 0):
            note(k.off)

    def walk(el, parent=None, idx=0):
        check(el, parent, idx)
        for i, k in enumerate(el.kids):
            walk(k, el, i)
        key = (el.val_off, el.val_off + el.val_len)
        if (not el.cons and el.cls == UNIV and el.num == 4 and key in pset
                and el.val_len > 0):
            try:
                walk(parse_el(d, el.val_off, len(d)))
            except (IndexError, ValueError):
                pass

    walk(root)
    if bad[0] is None:
        return "accept", None
    return "reject", bad[0]


def recipe_expectation(quirks, reg):
    """The offset the recipe itself predicts, or None."""
    worst = None
    for loc, (node, region) in reg.items():
        mode = quirks[loc]
        if isinstance(mode, int) or mode == "unsorted":
            kind = "unsorted"
        else:
            kind = mode
        ok = (kind != "unsorted") if region == "CERT" else (kind in ("indef", "unsorted"))
        if not ok and (worst is None or node.off < worst):
            worst = node.off
    return worst


# ------------------------------------------------------------- the case lists

CERT_CASES = [
    ("cert-canonical", {}),
    ("cert-issuer-rdn-unsorted", {"cert.issuer": 1}),
    ("cert-subject-rdn-unsorted", {"cert.subject": 2}),
    ("cert-both-names-unsorted", {"cert.issuer": 1, "cert.subject": 2}),
    ("cert-unsorted-rdn-then-serial-pad", {"cert.issuer": 1, "cert.serial": "pad"}),
    ("cert-serial-pad", {"cert.serial": "pad"}),
    ("cert-sigalg-pad", {"cert.sigalg": "pad"}),
    ("cert-sigalg-indefinite", {"cert.sigalg": "indef"}),
    ("cert-tbs-indefinite", {"cert.tbs": "indef"}),
    ("cert-validity-pad", {"cert.validity": "pad"}),
    ("cert-name-pad", {"cert.issuername": "pad"}),
    ("cert-spki-constructed-bitstring", {"cert.spki": "cons"}),
    ("cert-sigvalue-constructed-bitstring", {"cert.sigvalue": "cons"}),
    ("cert-sigvalue-pad", {"cert.sigvalue": "pad"}),
    ("cert-extwrap-indefinite", {"cert.extswrap": "indef"}),
    ("cert-critical-explicit-false", {"cert.critical": "default"}),
    ("cert-extvalue-constructed-octetstring", {"cert.extvalue": "cons"}),
    ("cert-subject-rdn-unsorted-other-index", {"cert.subject": 1}),
    ("cert-issuer-rdn-unsorted-plus-validity-pad",
     {"cert.issuer": 2, "cert.validity": "pad"}),
    ("cert-sigalg-pad-plus-spki-cons", {"cert.sigalg": "pad", "cert.spki": "cons"}),
]

LIB_CASES = [
    ("cms-canonical", {}),
    ("cms-content-pad", {"cms.content": "pad"}),
    ("cms-content-indefinite", {"cms.content": "indef"}),
    ("cms-sd-version-pad", {"cms.sdversion": "pad"}),
    ("cms-sd-indefinite", {"cms.sd": "indef"}),
    ("cms-digest-algs-pad", {"cms.digests": "pad"}),
    ("cms-digest-algs-indefinite", {"cms.digests": "indef"}),
    ("cms-digest-algs-unsorted", {"cms.digests": "unsorted"}),
    ("cms-certificates-pad", {"cms.certs": "pad"}),
    ("cms-certificates-indefinite", {"cms.certs": "indef"}),
    ("cms-certificates-unsorted", {"cms.certs": "unsorted"}),
    ("cms-si-sigalg-pad", {"cms.si.sigalg": "pad"}),
    ("cms-si-sid-serial-pad", {"cms.si.sid.serial": "pad"}),
    ("cms-si-sid-name-rdn-unsorted", {"cms.si.sid.name": 1}),
    ("cms-si-sid-name-pad", {"cms.si.sid.namepad": "pad"}),
    ("cms-si-sid-name-indefinite", {"cms.si.sid.namepad": "indef"}),
    ("cms-signedattrs-pad", {"cms.si.attrs": "pad"}),
    ("cms-signedattrs-indefinite", {"cms.si.attrs": "indef"}),
    ("cms-signedattrs-unsorted", {"cms.si.attrs": "unsorted"}),
    ("cms-signerinfos-pad", {"cms.signerinfos": "pad"}),
    ("cms-signerinfos-indefinite", {"cms.signerinfos": "indef"}),
    ("cms-signerinfos-unsorted", {"cms.signerinfos": "unsorted"}),
    ("cms-signature-constructed-octetstring", {"cms.si.signature": "cons"}),
    ("cms-signature-pad", {"cms.si.signature": "pad"}),
    ("cms-messagedigest-value-constructed", {"cms.si.attr.md.value": "cons"}),
    ("ext-basicconstraints-ca-explicit-false", {"ext.inner.bc.ca": "default"}),
    ("ext-basicconstraints-pad", {"ext.inner.bc.seq": "pad"}),
    ("ext-basicconstraints-indefinite", {"ext.inner.bc.seq": "indef"}),
    ("ext-basicconstraints-pathlen-pad", {"ext.inner.bc.pathlen": "pad"}),
    ("ext-keyusage-constructed-bitstring", {"ext.inner.ku": "cons"}),
    ("ext-subjectkeyid-constructed-octetstring", {"ext.inner.ski": "cons"}),
    ("ext-authkeyid-pad", {"ext.inner.aki": "pad"}),
    ("ext-extkeyusage-pad", {"ext.inner.eku": "pad"}),
    ("ext-extkeyusage-indefinite", {"ext.inner.eku": "indef"}),
    ("ext-private-set-unsorted", {"ext.inner.custom.set": "unsorted"}),
    ("ext-private-seq-indefinite", {"ext.inner.custom.seq": "indef"}),
]

MIXED_CASES = [
    ("mix-sid-name-rdn-unsorted-alone", {"cms.si.sid.name": 2}),
    ("mix-issuer-unsorted-and-sid-unsorted",
     {"cert.issuer": 1, "cms.si.sid.name": 2}),
    ("mix-name-pad-cert-and-sid", {"cert.issuername": "indef", "cms.si.sid.namepad": "pad"}),
    ("mix-content-pad-and-cert-unsorted", {"cms.content": "pad", "cert.issuer": 1}),
    ("mix-certwrapper-pad-and-cert-unsorted", {"cms.certs": "pad", "cert.issuer": 1}),
    ("mix-subject-unsorted-and-signature-constructed",
     {"cert.subject": 2, "cms.si.signature": "cons"}),
    ("mix-inner-default-late-and-subject-unsorted-early",
     {"cert.subject": 1, "ext.inner.bc.ca": "default"}),
    ("mix-inner-default-early-and-sig-constructed-late",
     {"ext.inner.bc.ca": "default", "cms.si.signature": "cons"}),
    ("mix-critical-default-and-inner-default",
     {"cert.critical": "default", "ext.inner.bc.ca": "default"}),
    ("mix-serial-pad-and-inner-pad", {"cert.serial": "pad", "ext.inner.bc.seq": "pad"}),
    ("mix-two-library-violations", {"cms.digests": "pad", "cms.si.signature": "cons"}),
    ("mix-digests-unsorted-and-sid-unsorted",
     {"cms.digests": "unsorted", "cms.si.sid.name": 1}),
    ("mix-indefinite-everywhere",
     {"cms.content": "indef", "cert.tbs": "indef", "ext.inner.bc.seq": "indef"}),
    ("mix-constructed-strings-both-regions", {"cert.spki": "cons", "ext.inner.ku": "cons"}),
    ("mix-everything-tolerated",
     {"cms.content": "indef", "cms.digests": "unsorted", "cms.si.attrs": "unsorted",
      "cms.si.sid.name": 1, "cert.tbs": "indef", "cert.serial": "pad",
      "cert.sigvalue": "cons", "cert.critical": "default",
      "ext.inner.custom.set": "unsorted", "ext.inner.bc.seq": "indef"}),
    ("mix-three-rejections-pick-smallest",
     {"cms.content": "pad", "cert.issuer": 1, "cms.si.signature": "cons"}),
]


# Constructs that the probe corpus carries but the graded corpus does not.
# A constructed extnValue is the one place where "read by the certificate
# reader" and "parsed by the library" can be argued either way; the visible
# corpus keeps it so a solver can settle it against the binary, but it is not
# graded.
PROBE_ONLY = ("cert-extvalue-constructed-octetstring",)


def graded(cases):
    return [(l, q) for (l, q) in cases if l not in PROBE_ONLY]


def build_record(idx, name, label, quirks):
    if "ext.inner.bc.pathlen" in quirks:
        si = idx % 2
    else:
        si = idx % len(CERT_SPECS)
    spec = CERT_SPECS[si]
    helper = CERT_SPECS[0] if si != 0 else CERT_SPECS[1]
    reg = {}
    root = build_object(spec, helper, quirks, reg)
    data = bytes(enc(root, 0))
    fix_deferred()
    can = canon(root)
    if can is None:
        raise SystemExit("object %s canonicalised away" % name)
    verdict, off = model(data)
    exp = recipe_expectation(quirks, reg)
    if verdict == "accept" and exp is not None:
        raise SystemExit("recipe/model drift on %s (%s): expect %s" % (name, label, exp))
    if verdict == "reject" and exp != off:
        raise SystemExit("recipe/model drift on %s (%s): %r vs %r"
                         % (name, label, exp, off))
    return dict(object=name, label=label, quirks=quirks, cert=si, data=data,
                canonical=can, verdict=verdict, offset=off)


def build_split(prefix, cases, count):
    objs = []
    for i in range(count):
        label, quirks = cases[i % len(cases)]
        name = "%s%04d.der" % (prefix, i + 1)
        objs.append(build_record(i, name, label, dict(quirks)))
    return objs


def pem(der):
    b = base64.b64encode(der).decode()
    body = "\n".join(b[i:i + 64] for i in range(0, len(b), 64))
    return "-----BEGIN CERTIFICATE-----\n" + body + "\n-----END CERTIFICATE-----\n"


def write_split(out, name, objs, trust=False):
    base = os.path.join(out, name)
    odir = os.path.join(base, "corpus")
    os.makedirs(odir, exist_ok=True)
    for o in objs:
        with open(os.path.join(odir, o["object"]), "wb") as f:
            f.write(o["data"])
    with open(os.path.join(base, "reference.ndjson"), "w") as f:
        for o in objs:
            f.write(json.dumps({
                "object": o["object"],
                "canonical_der": base64.b64encode(o["canonical"]).decode(),
                "verdict": o["verdict"],
                "first_rejecting_element": o["offset"],
            }) + "\n")
    if trust:
        tdir = os.path.join(base, "trust_store")
        os.makedirs(tdir, exist_ok=True)
        seen = set()
        n = 0
        for i, spec in enumerate(CERT_SPECS):
            c = build_cert(spec, {}, {})
            der = bytes(enc(c, 0))
            fix_deferred()
            if der in seen:
                continue
            seen.add(der)
            n += 1
            with open(os.path.join(tdir, "trust%02d.pem" % n), "w") as f:
                f.write(pem(der))


def write_audit_log(out, objs):
    rows = []
    for o in objs:
        if o.get("logged"):
            rows.append({"object": o["object"], "verdict": o["verdict"]})
    rows.sort(key=lambda r: r["object"])
    path = os.path.join(out, "visible", "audit_log.ndjson")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def cross_check(accepts, objs, tag):
    bad = 0
    for o in objs:
        tmp = os.path.join("/tmp", "ccheck.der")
        with open(tmp, "wb") as f:
            f.write(o["data"])
        p = subprocess.run([accepts, tmp], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)
        out = p.stdout.decode().strip().split()
        if not out:
            raise SystemExit("accepts produced nothing for %s" % o["object"])
        v = out[0]
        off = int(out[1]) if len(out) > 1 else None
        if v != o["verdict"] or off != o["offset"]:
            bad += 1
            print("MISMATCH %s %s: binary=%s/%s reference=%s/%s"
                  % (tag, o["object"], v, off, o["verdict"], o["offset"]),
                  file=sys.stderr)
    if bad:
        raise SystemExit("%d binary/reference mismatches in the %s split" % (bad, tag))
    print("accepts reproduces the reference for all %d %s objects" % (len(objs), tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="all", choices=("visible", "heldout", "all"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--accepts")
    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)

    cert_objs = build_split("o", CERT_CASES, 20)
    lib_objs = build_split("o", LIB_CASES, 20)
    for o in cert_objs:
        o["logged"] = True
    for o in lib_objs:
        o["logged"] = False
    visible = cert_objs + lib_objs
    order = list(range(len(visible)))
    random.Random(args.seed ^ 0x5A5A).shuffle(order)
    visible = [visible[i] for i in order]
    for i, o in enumerate(visible):
        o["object"] = "o%04d.der" % (i + 1)
    held_cases = graded(MIXED_CASES + LIB_CASES + CERT_CASES)
    held = build_split("h", held_cases, len(held_cases))
    for i, o in enumerate(held):
        o["object"] = "h%04d.der" % (i + 1)

    if args.split in ("visible", "all"):
        write_split(out, "visible", visible, trust=True)
        n = write_audit_log(out, visible)
        print("visible: %d objects, %d audit-log entries" % (len(visible), n))
    if args.split in ("heldout", "all"):
        write_split(out, "heldout", held)
        print("heldout: %d objects" % len(held))

    if args.accepts:
        if args.split in ("visible", "all"):
            cross_check(args.accepts, visible, "visible")
        if args.split in ("heldout", "all"):
            cross_check(args.accepts, held, "heldout")


if __name__ == "__main__":
    main()
