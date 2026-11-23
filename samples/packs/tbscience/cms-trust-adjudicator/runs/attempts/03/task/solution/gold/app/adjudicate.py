#!/usr/bin/env python3
"""Canonicalise and adjudicate BER/DER CMS SignedData objects.

No cryptographic operation is required: this reproduces the appliance's
encoding-policy pass, not signature verification.
"""
import argparse, base64, json, os

class ASN1Error(Exception): pass

class N:
    __slots__=('data','start','content','end','cls','tag','cons','indef','badlen','children','parent','role','embedded','omit')
    def __init__(self,d,s,c,e,cl,t,co,ind,bad,ch):
        self.data=d; self.start=s; self.content=c; self.end=e
        self.cls=cl; self.tag=t; self.cons=co; self.indef=ind; self.badlen=bad
        self.children=ch; self.parent=None; self.role=None; self.embedded=None; self.omit=False
        for x in ch: x.parent=self
    @property
    def raw_content(self): return self.data[self.content:self.end-(2 if self.indef else 0)]
    @property
    def full(self): return self.data[self.start:self.end]

def one(d,p,limit):
    s=p
    if p>=limit: raise ASN1Error('missing tag')
    b=d[p]; p+=1; cl=b&0xc0; co=bool(b&0x20); t=b&31
    if t==31:
        t=0
        if p>=limit: raise ASN1Error('tag')
        while True:
            q=d[p]; p+=1; t=(t<<7)|(q&127)
            if not q&128: break
            if p>=limit: raise ASN1Error('tag')
    if p>=limit: raise ASN1Error('length')
    q=d[p]; p+=1; ind=False; bad=False
    if q==128:
        if not co: raise ASN1Error('primitive indefinite')
        ind=True; stop=None
    elif q<128: stop=p+q
    else:
        n=q&127
        if n==0 or n>8 or p+n>limit: raise ASN1Error('length')
        z=d[p:p+n]; p+=n
        ln=int.from_bytes(z,'big')
        bad=(ln<=127 or z[0]==0)
        stop=p+ln
        if stop>limit: raise ASN1Error('overrun')
    content=p; ch=[]
    if co:
        if ind:
            while True:
                if p+2>limit: raise ASN1Error('eoc')
                if d[p:p+2]==b'\0\0': p+=2; break
                x,p=one(d,p,limit); ch.append(x)
            e=p
        else:
            while p<stop:
                x,p=one(d,p,stop); ch.append(x)
            if p!=stop: raise ASN1Error('contents')
            e=stop
    else: e=stop
    return N(d,s,content,e,cl,t,co,ind,bad,ch),e

def parse(d, base=0):
    n,e=one(d,0,len(d))
    if e!=len(d): raise ASN1Error('trailing data')
    if base: shift(n,base,None)
    return n

def shift(n,k,outer):
    n.start+=k; n.content+=k; n.end+=k
    if outer is not None: n.data=outer
    for c in n.children: shift(c,k,outer)

def tagbytes(cl,cons,t):
    b=cl | (0x20 if cons else 0)
    if t<31:return bytes((b|t,))
    a=[]
    while t: a.append(t&127); t>>=7
    if not a:a=[0]
    a.reverse()
    for i in range(len(a)-1): a[i]|=128
    return bytes((b|31,*a))
def length(n):
    if n<128:return bytes((n,))
    b=n.to_bytes((n.bit_length()+7)//8,'big'); return bytes((128|len(b),))+b
def tlv(cl,cons,t,c): return tagbytes(cl,cons,t)+length(len(c))+c

def oid(n):
    return n.raw_content if n.cls==0 and n.tag==6 else None
# DER content octets for common OIDs
OID_SIGNED=b'\x2a\x86\x48\x86\xf7\x0d\x01\x07\x02'
# X.509 extension OIDs whose extnValue is one ASN.1 value. All X.509
# extensions have this syntax; unknown private extensions in this appliance do too.

def mark_schema(root):
    """Attach abstract-type facts hidden by tags and locate reader domains."""
    cert_ranges=[]; tolerant_ranges=[]
    # Universal SET is SET OF in every structure used here (RDN included).
    def walk(n):
        if n.cls==0 and n.tag==17: n.role='setof'
        for c in n.children: walk(c)
    walk(root)
    try:
        # ContentInfo ::= SEQUENCE {oid, [0] EXPLICIT SignedData}
        sd=root.children[1].children[0]
        if oid(root.children[0])!=OID_SIGNED: return cert_ranges,tolerant_ranges
        kids=sd.children
        if len(kids)>1: kids[1].role='setof'       # digestAlgorithms
        i=3
        if i<len(kids) and kids[i].cls==0x80 and kids[i].tag==0:
            kids[i].role='setof'                    # certificates IMPLICIT SET OF
            for cert in kids[i].children:
                # CertificateChoices: ordinary Certificate is a SEQUENCE.
                if cert.cls==0 and cert.tag==16:
                    cert_ranges.append((cert.start,cert.end))
                    mark_certificate(cert)
            i+=1
        if i<len(kids) and kids[i].cls==0x80 and kids[i].tag==1:
            kids[i].role='setof'; i+=1
        if i<len(kids):
            kids[i].role='setof'                    # signerInfos
            for si in kids[i].children: mark_signer(si)
    except (IndexError,AttributeError): pass
    # Embedded extension values are traversed by the appliance's second reader.
    def collect(n):
        if n.embedded:
            tolerant_ranges.append((n.content,n.content+len(n.raw_content)))
        for c in n.children: collect(c)
    collect(root)
    return cert_ranges,tolerant_ranges

def mark_signer(si):
    if not si.cons:return
    # SignedAttrs and UnsignedAttrs are IMPLICIT SET OF Attribute.
    for x in si.children:
        if x.cls==0x80 and x.tag in (0,1): x.role='setof'

def mark_certificate(cert):
    try:
        tbs=cert.children[0]; k=tbs.children
        # TBSCertificate.version is [0] EXPLICIT Version DEFAULT v1(0).
        if k and k[0].cls==0x80 and k[0].tag==0 and k[0].children:
            v=k[0].children[0]
            if v.cls==0 and v.tag==2 and int.from_bytes(v.raw_content,'big',signed=True)==0:
                k[0].omit=True
        # Extensions are [3] EXPLICIT Extensions.  extnValue contains one
        # ASN.1 value and is therefore canonicalised recursively.
        for x in k:
            if x.cls==0x80 and x.tag==3 and x.cons:
                seq=x.children[0]
                for ext in seq.children:
                    ek=ext.children
                    if len(ek)>=2:
                        # Extension.critical BOOLEAN DEFAULT FALSE.
                        if len(ek)>=3 and ek[1].cls==0 and ek[1].tag==1 and not any(ek[1].raw_content):
                            ek[1].omit=True
                        val=ek[-1]
                        if val.cls==0 and val.tag==4 and not val.cons:
                            try:
                                val.embedded=parse(val.raw_content); shift(val.embedded,val.content,val.data)
                                # BasicConstraints.cA BOOLEAN DEFAULT FALSE.
                                if oid(ek[0])==b'\x55\x1d\x13' and val.embedded.children:
                                    q=val.embedded.children[0]
                                    if q.cls==0 and q.tag==1 and not any(q.raw_content): q.omit=True
                            except ASN1Error: pass
    except (IndexError,AttributeError): pass

def flatten_octets(n):
    if not n.cons:return n.raw_content
    out=[]
    def f(x):
        if x.cls!=0 or x.tag!=4: raise ASN1Error('octet segment')
        if x.cons:
            for y in x.children:f(y)
        else:out.append(x.raw_content)
    f(n); return b''.join(out)
def flatten_bits(n):
    if not n.cons:return n.raw_content
    seg=[]
    def f(x):
        if x.cls!=0 or x.tag!=3: raise ASN1Error('bit segment')
        if x.cons:
            for y in x.children:f(y)
        else:seg.append(x.raw_content)
    f(n)
    if not seg:return b'\0'
    out=bytearray((seg[-1][0],))
    for i,s in enumerate(seg):
        if not s: raise ASN1Error('bit segment')
        if i+1<len(seg) and s[0]!=0: raise ASN1Error('bit segment')
        out.extend(s[1:])
    return bytes(out)
def minint(b):
    if not b:return b'\0'
    while len(b)>1 and ((b[0]==0 and b[1]<128) or (b[0]==255 and b[1]>=128)): b=b[1:]
    return b

def canonical(n):
    # Strings required primitive by DER.
    if n.cls==0 and n.tag==4:
        c=flatten_octets(n)
        if n.embedded:
            # Reparse against the flattened bytes if construction changed offsets.
            try: c=canonical(n.embedded)
            except ASN1Error: pass
        return tlv(0,False,4,c)
    if n.cls==0 and n.tag==3:
        return tlv(0,False,3,flatten_bits(n))
    if n.cons:
        members=[]
        for c in n.children:
            if c.omit: continue
            members.append(canonical(c))
        if n.role=='setof' or (n.cls==0 and n.tag==17): members.sort()
        return tlv(n.cls,True,n.tag,b''.join(members))
    c=n.raw_content
    if n.cls==0 and n.tag==2:c=minint(c)
    elif n.cls==0 and n.tag==1:c=b'\xff' if c and any(c) else b'\0'
    return tlv(n.cls,False,n.tag,c)

def der_len_ok(n): return not n.badlen

def integer_ok(n): return bool(n.raw_content) and minint(n.raw_content)==n.raw_content
def bool_ok(n): return len(n.raw_content)==1 and n.raw_content in (b'\0',b'\xff')
def set_order_ok(n):
    vals=[c.full for c in n.children]
    return all(vals[i-1]<vals[i] for i in range(1,len(vals)))

def verdict(root,cert_ranges,tolerant_ranges):
    bad=[]
    def inside(pos,ranges): return any(a<=pos<b for a,b in ranges)
    def rec(n):
        cert=inside(n.start,cert_ranges)
        tol=inside(n.start,tolerant_ranges)
        # Certificate reader ignores every canonicality property except SET order.
        # Embedded-value reader likewise checks ordering but tolerates BER forms.
        if cert and not tol:
            if n.cls==0 and n.tag==17 and n.cons and not set_order_ok(n): bad.append(n.start)
        else:
            if not der_len_ok(n): bad.append(n.start)
            if n.cls==0 and n.tag in (3,4) and n.cons: bad.append(n.start)
            if n.cls==0 and n.tag==2 and not integer_ok(n): bad.append(n.start)
            if n.cls==0 and n.tag==1 and len(n.raw_content)==1 and n.raw_content==b'\0': bad.append(n.start)
        for c in n.children:rec(c)
        if n.embedded:
            # Its shifted offsets point into the outer input.
            rec(n.embedded)
    rec(root)
    return min(bad) if bad else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--corpus',required=True); a=ap.parse_args()
    for name in sorted(x for x in os.listdir(a.corpus) if x.endswith('.der')):
        with open(os.path.join(a.corpus,name),'rb') as f:d=f.read()
        root=parse(d); cr,tr=mark_schema(root); off=verdict(root,cr,tr)
        out={'object':name,'canonical_der':base64.b64encode(canonical(root)).decode('ascii'),
             'verdict':'reject' if off is not None else 'accept','first_rejecting_element':off}
        print(json.dumps(out,separators=(',',':')))
if __name__=='__main__':main()
