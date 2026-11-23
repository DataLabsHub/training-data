/*
 * accepts -- report this appliance's verdict for one CMS ContentInfo object.
 *
 *   accepts FILE
 *     prints "accept", or "reject <byte-offset>", on stdout.
 *     exit status 0 when the object is accepted, 1 when it is rejected,
 *     2 on a usage or I/O error.
 *
 * The walker is table driven: an element's tolerance is decided by the parse
 * path that reaches it, not by the element alone.  Elements reached through an
 * embedded Certificate -- other than the body of an extnValue OCTET STRING --
 * are read by the hand rolled certificate reader; everything else, including
 * the certificates [0] wrapper itself, the CMS envelope, the signed attributes
 * and the extension inner values, is read by the linked CMS library.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CERTS 16
#define MAX_EVALS 128

typedef struct {
    size_t off;      /* offset of the tag octet               */
    size_t val_off;  /* offset of the first content octet     */
    size_t val_len;  /* content length (indefinite: computed) */
    size_t end;      /* offset just past the element          */
    int cls;         /* tag class bits                        */
    int cons;        /* constructed bit                       */
    unsigned long num; /* tag number                          */
    int indef;       /* length octets were 80 ... 00 00       */
    int len_min;     /* length octets are the minimal form    */
} Hdr;

typedef struct {
    const unsigned char *b;
    size_t n;
    size_t bad;
    int nbad;
} Ctx;

/* ----------------------------------------------------------------- ranges */

static size_t cert_a[MAX_CERTS], cert_b[MAX_CERTS];
static int n_cert;
static size_t eval_a[MAX_EVALS], eval_b[MAX_EVALS];
static int n_eval;

/* ------------------------------------------------------------ TLV scanning */

static int parse_hdr(const unsigned char *b, size_t n, size_t p, Hdr *h)
{
    size_t q = p;
    int first;

    if (p >= n)
        return -1;
    first = b[q++];
    h->off = p;
    h->cls = first & 0xC0;
    h->cons = (first & 0x20) ? 1 : 0;
    h->num = (unsigned long)(first & 0x1F);

    if (h->num == 0x1F) {
        unsigned long v = 0;
        int cnt = 0;
        for (;;) {
            if (q >= n || cnt >= 6)
                return -1;
            v = (v << 7) | (unsigned long)(b[q] & 0x7F);
            cnt++;
            if (!(b[q++] & 0x80))
                break;
        }
        h->num = v;
    }

    if (q >= n)
        return -1;
    {
        unsigned char l = b[q++];
        h->indef = 0;
        h->len_min = 1;
        if (l == 0x80) {
            h->indef = 1;
            h->val_len = 0;
        } else if (l < 0x80) {
            h->val_len = (size_t)l;
        } else {
            int k = l & 0x7F;
            size_t v = 0;
            int i;
            if (k == 0 || k > 8 || q + (size_t)k > n)
                return -1;
            for (i = 0; i < k; i++)
                v = (v << 8) | (size_t)b[q + i];
            h->len_min = (v >= 0x80 && b[q] != 0) ? 1 : 0;
            q += (size_t)k;
            h->val_len = v;
        }
    }

    h->val_off = q;
    if (h->indef) {
        size_t r = q;
        for (;;) {
            Hdr c;
            if (r + 2 > n)
                return -1;
            if (b[r] == 0x00 && b[r + 1] == 0x00) {
                r += 2;
                break;
            }
            if (parse_hdr(b, n, r, &c) != 0)
                return -1;
            r = c.end;
        }
        h->val_len = r - 2 - q;
        h->end = r;
    } else {
        if (q + h->val_len > n)
            return -1;
        h->end = q + h->val_len;
    }
    return 0;
}

static int is_eoc(const Hdr *h)
{
    return h->cls == 0 && h->num == 0 && h->val_len == 0 && !h->cons;
}

static size_t content_end(const Hdr *h)
{
    return h->indef ? (h->end - 2) : (h->val_off + h->val_len);
}

static int nth_child(const unsigned char *b, size_t n, size_t begin,
                     size_t end, int want, Hdr *out)
{
    size_t q = begin;
    int i = 0;
    while (q < end) {
        Hdr h;
        if (parse_hdr(b, n, q, &h) != 0)
            return -1;
        if (is_eoc(&h))
            break;
        if (i == want) {
            *out = h;
            return 0;
        }
        q = h.end;
        i++;
    }
    return -1;
}

static int last_child(const unsigned char *b, size_t n, size_t begin,
                      size_t end, Hdr *out)
{
    size_t q = begin;
    int got = -1;
    while (q < end) {
        Hdr h;
        if (parse_hdr(b, n, q, &h) != 0)
            return -1;
        if (is_eoc(&h))
            break;
        *out = h;
        got = 0;
        q = h.end;
    }
    return got;
}

/* ------------------------------------------------------------- range lookup */

static int in_eval(size_t off)
{
    int i;
    for (i = 0; i < n_eval; i++)
        if (off >= eval_a[i] && off < eval_b[i])
            return 1;
    return 0;
}

static int in_cert(size_t off)
{
    int i;
    for (i = 0; i < n_cert; i++)
        if (off >= cert_a[i] && off < cert_b[i])
            return 1;
    return 0;
}

/* 1 = the hand rolled certificate reader, 0 = the linked CMS library. */
static int cert_reader_owns(size_t off)
{
    if (in_eval(off))
        return 0;
    return in_cert(off);
}

/* ------------------------------------------------------------------- table */

/* Everywhere the library parses, these constructs are not tolerated. */
static void note(Ctx *c, size_t off)
{
    if (!c->nbad || off < c->bad) {
        c->bad = off;
        c->nbad = 1;
    }
}

static int set_sorted(const Ctx *c, const Hdr *h)
{
    size_t q = h->val_off, prev = 0, prevlen = 0;
    int first = 1;
    if (!h->cons)
        return 1;
    while (q < content_end(h)) {
        Hdr k;
        size_t len;
        if (parse_hdr(c->b, c->n, q, &k) != 0)
            return 1;
        if (is_eoc(&k))
            break;
        len = k.end - k.off;
        if (!first) {
            size_t m = prevlen < len ? prevlen : len;
            int cmp = memcmp(c->b + prev, c->b + k.off, m);
            if (cmp > 0 || (cmp == 0 && prevlen >= len))
                return 0;
        }
        prev = k.off;
        prevlen = len;
        first = 0;
        q = k.end;
    }
    return 1;
}

static int int_minimal(const Ctx *c, const Hdr *h)
{
    const unsigned char *v = c->b + h->val_off;
    if (h->val_len == 0)
        return 0;
    if (h->val_len >= 2) {
        if (v[0] == 0x00 && v[1] < 0x80)
            return 0;
        if (v[0] == 0xFF && v[1] >= 0x80)
            return 0;
    }
    return 1;
}

static void check_el(Ctx *c, const Hdr *h, unsigned long ptag, int idx)
{
    if (cert_reader_owns(h->off)) {
        /* The certificate reader tolerates everything but a SET OF out of order. */
        if (h->cls == 0 && h->cons && h->num == 17 && !set_sorted(c, h))
            note(c, h->off);
        return;
    }
    if (!h->len_min)
        note(c, h->off);
    if (h->cls == 0 && h->cons && (h->num == 3 || h->num == 4))
        note(c, h->off);
    if (h->cls == 0 && !h->cons && h->num == 2 && !int_minimal(c, h))
        note(c, h->off);
    if (h->cls == 0 && !h->cons && h->num == 1 && ptag == 16 && idx == 0
        && h->val_len == 1 && c->b[h->val_off] == 0x00)
        note(c, h->off);
}

/* ------------------------------------------------------------------ walking */

static void walk_deep(Ctx *c, const Hdr *parent, size_t begin, size_t stop);

static void walk_top(Ctx *c, size_t begin, size_t stop)
{
    Hdr h;
    if (parse_hdr(c->b, c->n, begin, &h) != 0)
        return;
    check_el(c, &h, 0xFFFFFFFFUL, 0);
    if (h.cons)
        walk_deep(c, &h, h.val_off, content_end(&h));
}

static void walk_deep(Ctx *c, const Hdr *parent, size_t begin, size_t stop)
{
    size_t q = begin;
    unsigned long ptag = 0xFFFFFFFFUL;
    int idx = 0;

    if (parent->cls == 0)
        ptag = parent->num;

    while (q < stop) {
        Hdr h;
        if (parse_hdr(c->b, c->n, q, &h) != 0)
            return;
        if (is_eoc(&h))
            return;
        check_el(c, &h, ptag, idx);
        if (h.cons)
            walk_deep(c, &h, h.val_off, content_end(&h));
        if (!h.cons && h.cls == 0 && h.num == 4 && h.val_len > 0
            && in_eval(h.val_off) && in_eval(h.val_off + h.val_len - 1))
            walk_top(c, h.val_off, h.val_off + h.val_len);
        q = h.end;
        idx++;
    }
}

/* ------------------------------------------------------------- range finding */

static void find_ranges(Ctx *c)
{
    Hdr root, wrap, sd, h, tbs, ext, list, ev;
    size_t q, stop;
    int i;

    n_cert = 0;
    n_eval = 0;

    if (parse_hdr(c->b, c->n, 0, &root) != 0 || !root.cons)
        return;
    if (nth_child(c->b, c->n, root.val_off, content_end(&root), 1, &wrap) != 0)
        return;
    if (!wrap.cons)
        return;
    if (nth_child(c->b, c->n, wrap.val_off, content_end(&wrap), 0, &sd) != 0)
        return;
    if (!sd.cons)
        return;

    q = sd.val_off;
    stop = content_end(&sd);
    while (q < stop && n_cert < MAX_CERTS) {
        Hdr cs;
        if (parse_hdr(c->b, c->n, q, &cs) != 0)
            break;
        if (is_eoc(&cs))
            break;
        if (cs.cls == 0x80 && cs.num == 0 && cs.cons) {
            size_t p = cs.val_off, e = content_end(&cs);
            while (p < e && n_cert < MAX_CERTS) {
                Hdr ce;
                if (parse_hdr(c->b, c->n, p, &ce) != 0)
                    break;
                if (is_eoc(&ce))
                    break;
                cert_a[n_cert] = ce.off;
                cert_b[n_cert] = ce.end;
                n_cert++;
                p = ce.end;
            }
        }
        q = cs.end;
    }

    for (i = 0; i < n_cert; i++) {
        if (parse_hdr(c->b, c->n, cert_a[i], &h) != 0 || !h.cons)
            continue;
        if (nth_child(c->b, c->n, h.val_off, content_end(&h), 0, &tbs) != 0)
            continue;
        if (!tbs.cons)
            continue;
        q = tbs.val_off;
        stop = content_end(&tbs);
        while (q < stop) {
            Hdr k;
            if (parse_hdr(c->b, c->n, q, &k) != 0)
                break;
            if (is_eoc(&k))
                break;
            if (k.cls == 0x80 && k.num == 3 && k.cons) {
                size_t p, e2;
                if (nth_child(c->b, c->n, k.val_off, content_end(&k), 0, &list) != 0)
                    break;
                p = list.val_off;
                e2 = content_end(&list);
                while (p < e2 && n_eval < MAX_EVALS) {
                    Hdr ex;
                    if (parse_hdr(c->b, c->n, p, &ex) != 0)
                        break;
                    if (is_eoc(&ex))
                        break;
                    if (last_child(c->b, c->n, ex.val_off, content_end(&ex), &ev) == 0
                        && ev.cls == 0 && ev.num == 4) {
                        eval_a[n_eval] = ev.val_off;
                        eval_b[n_eval] = ev.val_off + ev.val_len;
                        n_eval++;
                    }
                    p = ex.end;
                }
            }
            q = k.end;
        }
    }
}

/* ---------------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    FILE *f;
    long sz;
    unsigned char *buf;
    Ctx c;

    if (argc != 2) {
        fprintf(stderr, "usage: %s FILE\n", argv[0]);
        return 2;
    }
    f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "accepts: cannot read %s\n", argv[1]);
        return 2;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 2;
    }
    sz = ftell(f);
    if (sz <= 0) {
        fclose(f);
        fprintf(stderr, "accepts: empty input\n");
        return 2;
    }
    rewind(f);
    buf = (unsigned char *)malloc((size_t)sz);
    if (!buf) {
        fclose(f);
        return 2;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        free(buf);
        return 2;
    }
    fclose(f);

    c.b = buf;
    c.n = (size_t)sz;
    c.bad = 0;
    c.nbad = 0;

    find_ranges(&c);
    walk_top(&c, 0, c.n);

    if (c.nbad) {
        printf("reject %lu\n", (unsigned long)c.bad);
        free(buf);
        return 1;
    }
    printf("accept\n");
    free(buf);
    return 0;
}
