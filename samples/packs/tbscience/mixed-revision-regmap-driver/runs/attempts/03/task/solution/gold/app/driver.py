#!/usr/bin/env python3
import argparse, csv, json, sys

# The C map, plus the placement established from the bench response for D.
REG_C = {
  0x00: [('PART_ID',12,4),('REV',8,4),('DIE_TRIM',0,8)],
  0x01: [('FAULT',8,8),('SEQ_STATE',0,8)],
  0x02: [('A_FREQ',0,8)], 0x03: [('A_GAIN',0,8)], 0x04: [('A_PHASE',0,8)],
  0x05: [('B_FREQ',0,8)], 0x06: [('B_GAIN',0,8)], 0x07: [('B_PHASE',0,8)],
  0x08: [('C_FREQ',0,8)], 0x09: [('C_GAIN',0,8)], 0x0a: [('C_PHASE',0,8)],
  0x0b: [('D_FREQ',0,8)], 0x0c: [('D_GAIN',0,8)], 0x0d: [('D_PHASE',0,8)],
  0x0e: [('EN_A',0,1),('EN_B',1,1),('EN_C',2,1),('EN_D',3,1),('CLKDIV',4,4)],
  0x0f: [('SCRATCH',0,16)]
}
REG_D = dict(REG_C)
REG_D[0x09] = [('C_GAIN',8,8),('D_GAIN',0,8)]
REG_D[0x0c] = []
CHANNELS = 'ABCD'

def read_bus(path):
    with open(path, newline='') as f:
        return [{**r, 't_us':int(r['t_us']), 'cs':int(r['cs']),
                 'addr':int(r['addr'],0), 'data':int(r['data'],0)}
                for r in csv.DictReader(f)]

def plateaus(path):
    vals=[]
    with open(path, newline='') as f:
        for r in csv.DictReader(f): vals.append(int(r['adc']))
    # Idle is near zero; every useful hold is hundreds of counts.  Ignore
    # short transition fragments and use a median, which rejects logger noise.
    runs=[]; start=None
    for i, v in enumerate(vals + [0]):
        if v > 100 and start is None: start=i
        elif v <= 100 and start is not None:
            if i-start >= 100:
                q=sorted(vals[start:i]); runs.append(q[len(q)//2])
            start=None
    return runs

def classify(rows, strain):
    # Each MASTER enable window is one channel.  Aggregate windows by the
    # CS encountered in the trace; neither CS values nor their order imply a
    # revision.
    segments=[]
    current=None
    for r in rows:
        if r['op']=='W' and r['addr']==0x0e and r['data'] in (1,2,4,8):
            if current is not None: segments.append(current)
            current={'cs':r['cs'], 'ch':r['data'].bit_length()-1, 'g':{}}
        elif current is not None and r['op']=='W' and r['addr'] in (0x03,0x06,0x09,0x0c):
            current['g'][r['addr']]=r['data']
    if current is not None: segments.append(current)
    if len(strain) < len(segments): raise ValueError('strain capture lacks scheduled holds')
    cards={}
    order=[]
    for i,seg in enumerate(segments):
        cs=seg['cs']
        if cs not in cards: cards[cs]={'cs':cs,'g':{},'levels':{}}; order.append(cs)
        c=cards[cs]
        c['levels'][seg['ch']]=strain[i]
        c['g'].update(seg['g'])
    # Shared A/B channels establish ADC = intercept + slope * gain.
    pairs=[]
    for cs in order:
        c=cards[cs]
        for ch,addr in ((0,0x03),(1,0x06)):
            if ch in c['levels']: pairs.append((c['g'].get(addr,0)&255,c['levels'][ch]))
    if len(pairs)>=2 and pairs[0][0] != pairs[-1][0]:
        g1,y1=pairs[0]; g2,y2=pairs[-1]
        slope=(y2-y1)/(g2-g1); intercept=y1-slope*g1
    else: slope,intercept=40.0,20.0
    result={}
    for cs in order:
        c=cards[cs]
        ec=[(c['levels'][ch]-intercept)/slope for ch in (2,3) if ch in c['levels']]
        old=[c['g'].get(0x09,0)&255,c['g'].get(0x0c,0)&255]
        new=[(c['g'].get(0x09,0)>>8)&255,c['g'].get(0x09,0)&255]
        errc=sum((ec[i]-old[i])**2 for i in range(min(2,len(ec))))
        errd=sum((ec[i]-new[i])**2 for i in range(min(2,len(ec))))
        result[cs]='D' if errd < errc else 'C'
    return result

def fields(addr,data,rev):
    table=REG_D if rev=='D' else REG_C
    return {n:(data >> lo) & ((1<<w)-1) for n,lo,w in table.get(addr,[])}

def do_decode(args):
    rows=read_bus(args.capture)
    revs=classify(rows,plateaus(args.strain))
    with open(args.out,'w',newline='\n') as f:
        for r in rows:
            obj={'t_us':r['t_us'],'cs':r['cs'],'op':r['op'],'addr':r['addr'],
                 'revision':revs[r['cs']], 'fields':fields(r['addr'],r['data'],revs[r['cs']])}
            f.write(json.dumps(obj,separators=(',',':'),sort_keys=False)+'\n')

def do_excite(args):
    with open(args.spec) as f: s=json.load(f)
    rev=s['revision']; cs=int(s['cs']); w=[]
    for ch in CHANNELS:
        w.append([cs,0x02+3*CHANNELS.index(ch),int(s['freq_hz'][ch])])
        w.append([cs,0x04+3*CHANNELS.index(ch),int(s['phase_deg'][ch])//2])
    if rev=='C':
        for ch in CHANNELS: w.append([cs,0x03+3*CHANNELS.index(ch),int(s['gain'][ch])])
    else:
        for ch in 'AB': w.append([cs,0x03+3*CHANNELS.index(ch),int(s['gain'][ch])])
        w.append([cs,0x09,(int(s['gain']['C'])<<8)|int(s['gain']['D'])])
    master=sum((int(s['enable'][ch]) & 1)<<i for i,ch in enumerate(CHANNELS))
    w.append([cs,0x0e,master])
    with open(args.out,'w',newline='\n') as f: json.dump({'writes':w},f,separators=(',',':')); f.write('\n')

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    d=sub.add_parser('decode'); d.add_argument('--capture',required=True); d.add_argument('--strain',required=True); d.add_argument('--out',required=True)
    e=sub.add_parser('excite'); e.add_argument('--spec',required=True); e.add_argument('--out',required=True)
    a=p.parse_args(); do_decode(a) if a.cmd=='decode' else do_excite(a)
if __name__=='__main__': main()
