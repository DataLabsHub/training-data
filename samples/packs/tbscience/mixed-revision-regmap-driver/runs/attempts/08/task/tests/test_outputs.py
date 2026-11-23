# CANARY: pd440-mixed-revision-regmap semantic verifier suite v001
"""Semantic verifier for security/mixed_revision_regmap_driver (v001).
Only /app/driver.py is graded; all references are recomputed in-suite from the
inlined true PD-440 model, never from a task file.
test_artifact      /app/driver.py exists and is non-empty.
test_stdlib_only   every import in /app/driver.py is stdlib.
test_decode        decode two held-out captures (mixed revisions with reserved
                   addresses and non-shipped card numbers/clock offset; all-rev-D
                   no-probe-read) and match the oracle keyed by the capture's
                   TRUE per-card revision, access-for-access and field-for-field;
                   revision comes from strain evidence, not the probe ID, and
                   literal anchors pin the probe word, rev-D 0x09 and rev-D 0x0C.
test_excite        replay the writes on a power-on die of the named revision:
                   every channel field exact, no read-only write.
test_deterministic decode and excite twice, byte-identical, unprivileged.
"""
import ast,json,os,shutil,subprocess,sys,tempfile
D="/app/driver.py";H="ABCD"
T={0:[("PART_ID",12,4),("REV",8,4),("DIE_TRIM",0,8)],1:[("FAULT",8,8),("SEQ_STATE",0,8)],
   14:[("EN_A",0,1),("EN_B",1,1),("EN_C",2,1),("EN_D",3,1),("CLKDIV",4,4)],15:[("SCRATCH",0,16)]}
for i,x in enumerate(H):T[2+3*i]=[(x+"_FREQ",0,8)];T[3+3*i]=[(x+"_GAIN",0,8)];T[4+3*i]=[(x+"_PHASE",0,8)]
U=dict(T);U[9]=[("C_GAIN",8,8),("D_GAIN",0,8)];U[12]=[]
def mk(v,a):
    if a in (0,1,15):return 0xFFFF
    if a==14:return 0xFF
    if v=="C":return 0xFF if 2<=a<=13 else 0
    if a==9:return 0xFFFF
    return 0xFF if (2<=a<=8 or a in (10,11,13)) else 0
class C:
    def __init__(s,v,t=0):s.v=v;s.t=t&255;s.g=[0]*128;s.r=15
    def w(s,a,d):
        if a>1 and a<128 and mk(s.v,a):s.g[a]=d&mk(s.v,a)
    def rd(s,a):
        if a==0:return (5<<12)|(3<<8)|s.t
        if a==1:return s.r
        return s.g[a]&mk(s.v,a) if a<128 else 0
    def gv(s,x):
        if x=="A":return s.g[3]&255
        if x=="B":return s.g[6]&255
        if x=="C":return (s.g[9]>>8)&255 if s.v=="D" else s.g[9]&255
        return s.g[9]&255 if s.v=="D" else s.g[12]&255
    def dv(s):return sum(14+40*s.gv(x) for x in H if (s.g[14]>>H.index(x))&1)
SW=[0x78,7,0x2D,0xC8,9,0x3C,0xF0,0x0A05,0x4B,0xFA,0x11,0x5A]
def gen(v,t,cs,off,pr=True,rs=False):
    cd={c:C(v[c],t[c]) for c in cs};R=[];Q=[0]
    def rd(c,a):R.append((Q[0],c,"R",a,cd[c].rd(a)));Q[0]+=8
    def wr(c,a,d):cd[c].w(a,d);R.append((Q[0],c,"W",a,d&0xFFFF));Q[0]+=8
    for c in cs:
        if pr:rd(c,0)
        wr(c,14,0)
        if rs:wr(c,0x20,0xBEEF);rd(c,0x30)
        for m in (1,2,4,8):
            wr(c,14,m)
            for r in range(2,14):wr(c,r,SW[r-2]);Q[0]+=320000
            wr(c,14,0);Q[0]+=320000
        rd(c,1);Q[0]+=1600000
    st={c:C(v[c],t[c]) for c in cs};sa=[];i=0;n=len(R);k=0;sd=0x5EED4402
    while k*4000<=Q[0]+off+16000:
        tb=k*4000-off
        while i<n and R[i][0]<=tb:
            _,c,o,a,d=R[i]
            if o=="W":st[c].w(a,d)
            i+=1
        sd=(sd*1103515245+12345)&0x7FFFFFFF
        sa.append((k*4000,6+sum(x.dv() for x in st.values())+(sd%13)-6));k+=1
    return R,sa
def orc(R,v):
    return [{"t_us":t,"cs":c,"op":o,"addr":a,"revision":v[c],
             "fields":{n:(d>>l)&((1<<w)-1) for n,l,w in (U if v[c]=="D" else T).get(a,[])}}
            for t,c,o,a,d in R]
def bcsv(R):return "t_us,cs,op,addr,data\n"+"".join("%d,%d,%s,0x%02X,0x%04X\n"%x for x in R)
def scsv(S):return "t_us,adc\n"+"".join("%d,%d\n"%x for x in S)
B=tempfile.mkdtemp(prefix="pd440_b_")
open(B+"/sitecustomize.py","w").write("import socket as s\ns.socket=s.create_connection=s.getaddrinfo=None\n")
os.chmod(B,0o755);os.chmod(B+"/sitecustomize.py",0o644)
if os.geteuid()==0:os.chmod(os.path.dirname(os.path.abspath(__file__)),0o700)
def sc():d=tempfile.mkdtemp(prefix="pd440_v_");os.chmod(d,0o777);return d
def put(d,n,x):p=d+"/"+n;open(p,"w",newline="\n").write(x);os.chmod(p,0o644);return p
def run(d,av):
    kw=dict(cwd=d,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=180,
            env={"PATH":os.environ.get("PATH","/usr/bin:/bin"),"HOME":"/tmp","PYTHONPATH":B})
    if os.geteuid()==0:kw["user"]="nobody"
    try:return subprocess.run([sys.executable,D]+av,**kw)
    except (OSError,ValueError,subprocess.SubprocessError):
        if os.geteuid()!=0:raise
        kw.pop("user",None);return subprocess.run([sys.executable,D]+av,**kw)
def need():
    assert os.path.isfile(D),"declared artifact /app/driver.py is missing"
    assert open(D).read().strip(),"/app/driver.py is empty"
    if os.geteuid()==0:
        try:os.chmod(os.path.dirname(D) or "/",0o755);os.chmod(D,0o755)
        except OSError:pass
CS=[("mixed",{4:"C",5:"D",6:"C",7:"D"},{4:0x11,5:0x22,6:0x33,7:0x44},[4,5,6,7],123456,True,True),
    ("all_d_noprobe",{1:"D",2:"D",3:"D",4:"D"},{1:1,2:2,3:3,4:4},[1,2,3,4],600000,False,False)]
SP=[("C",2,{"A":100,"B":150,"C":200,"D":250},{"A":5,"B":7,"C":9,"D":11},{"A":0,"B":90,"C":180,"D":270},{"A":1,"B":1,"C":0,"D":1}),
    ("D",5,{"A":11,"B":22,"C":33,"D":44},{"A":1,"B":63,"C":32,"D":7},{"A":2,"B":4,"C":6,"D":8},{"A":1,"B":0,"C":1,"D":1})]
def test_artifact():need()
def test_stdlib_only():
    need();bad=set()
    for n in ast.walk(ast.parse(open(D).read(),filename=D)):
        if isinstance(n,ast.Import):bad|={a.name.split(".")[0] for a in n.names if a.name.split(".")[0] not in sys.stdlib_module_names}
        elif isinstance(n,ast.ImportFrom):
            if n.level:bad.add("<relative>")
            elif n.module and n.module.split(".")[0] not in sys.stdlib_module_names:bad.add(n.module.split(".")[0])
    assert not bad,"non-stdlib imports: %s"%sorted(bad)
def test_decode():
    need()
    for cid,v,t,cs,off,pr,rs in CS:
        R,sa=gen(v,t,cs,off,pr,rs);ref=orc(R,v);d=sc()
        try:
            o=d+"/o.jsonl"
            r=run(d,["decode","--capture",put(d,"b.csv",bcsv(R)),"--strain",put(d,"s.csv",scsv(sa)),"--out",o])
            assert r.returncode==0,"%s rc=%d %s"%(cid,r.returncode,r.stderr[-1000:])
            got=[json.loads(x) for x in open(o).read().split("\n") if x.strip()]
            assert len(got)==len(ref),"%s emitted %d of %d"%(cid,len(got),len(ref))
            for i,(g,x) in enumerate(zip(got,ref)):
                assert set(g)=={"t_us","cs","op","addr","revision","fields"},"%s keys %s"%(cid,sorted(g))
                for k in ("t_us","cs","addr"):assert isinstance(g[k],int) and not isinstance(g[k],bool)
                assert g["op"] in ("R","W") and g["revision"] in ("C","D") and isinstance(g["fields"],dict)
                for z in g["fields"].values():assert isinstance(z,int) and not isinstance(z,bool)
                assert g==x,"%s #%d got %s want %s"%(cid,i,g,x)
            if cid=="mixed":
                ix={(g["cs"],g["op"],g["addr"]):g for g in got}
                assert ix[(4,"R",0)]["fields"]=={"PART_ID":5,"REV":3,"DIE_TRIM":0x11}
                assert ix[(5,"R",0)]["revision"]=="D" and ix[(5,"R",0)]["fields"]=={"PART_ID":5,"REV":3,"DIE_TRIM":0x22}
                assert ix[(4,"W",9)]["fields"]=={"C_GAIN":5} and ix[(5,"W",9)]["fields"]=={"C_GAIN":10,"D_GAIN":5}
                assert ix[(5,"W",12)]["fields"]=={} and ix[(4,"W",12)]["fields"]=={"D_GAIN":17}
        finally:shutil.rmtree(d,ignore_errors=True)
def test_excite():
    need()
    for rev,cs,fq,gn,ph,en in SP:
        d=sc()
        try:
            o=d+"/w.json";spec={"revision":rev,"cs":cs,"freq_hz":fq,"gain":gn,"phase_deg":ph,"enable":en}
            r=run(d,["excite","--spec",put(d,"p.json",json.dumps(spec)),"--out",o])
            assert r.returncode==0,"excite rc=%d %s"%(r.returncode,r.stderr[-1000:])
            doc=json.load(open(o));assert set(doc)=={"writes"} and doc["writes"]
            cr=C(rev)
            for w in doc["writes"]:
                assert isinstance(w,list) and len(w)==3 and all(isinstance(z,int) and not isinstance(z,bool) for z in w)
                assert w[0]==cs and 0<=w[1]<=0x7F and 0<=w[2]<=0xFFFF and w[1] not in (0,1),"bad write %s"%w
                cr.w(w[1],w[2])
            for i,x in enumerate(H):
                assert cr.g[2+3*i]&255==fq[x],"freq %s"%x
                assert (cr.g[4+3*i]&255)*2==ph[x],"phase %s"%x
                assert cr.gv(x)==gn[x],"gain %s"%x
                assert (cr.g[14]>>i)&1==(en[x]&1),"enable %s"%x
        finally:shutil.rmtree(d,ignore_errors=True)
def test_deterministic():
    need();_,v,t,cs,off,pr,rs=CS[0];R,sa=gen(v,t,cs,off,pr,rs);d=sc()
    try:
        b=put(d,"b.csv",bcsv(R));s=put(d,"s.csv",scsv(sa))
        rev,c2,fq,gn,ph,en=SP[0]
        p=put(d,"p.json",json.dumps({"revision":rev,"cs":c2,"freq_hz":fq,"gain":gn,"phase_deg":ph,"enable":en}))
        bl=[]
        for n in "ab":
            o1=d+"/d"+n+".jl";o2=d+"/w"+n+".json"
            assert run(d,["decode","--capture",b,"--strain",s,"--out",o1]).returncode==0
            assert run(d,["excite","--spec",p,"--out",o2]).returncode==0
            bl.append(open(o1,"rb").read()+b"\0"+open(o2,"rb").read())
        assert bl[0] and bl[0]==bl[1],"not byte-deterministic"
    finally:shutil.rmtree(d,ignore_errors=True)
