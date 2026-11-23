#!/usr/bin/env python3
"""Emit the pinned fixture sources for security/sbom-vs-linkage v002.

Builder-stage only.  Writes the five static-archive translation units, the
shim translation unit, two version-scripted shared objects, three plugin
DSOs and the service itself.  Nothing here ever reaches the shipped image;
the release bundle built from it by build_release.py is the only thing that
is COPYed into the final stage.
"""
import pathlib
import sys

SOURCES = {}

SOURCES["service.h"] = r"""
#ifndef FIXTURE_SERVICE_H
#define FIXTURE_SERVICE_H

#define ACME_SERVICE_VERSION "acme-service 3.7.0"

#endif
"""

SOURCES["hpack.h"] = r"""
#ifndef FIXTURE_HPACK_H
#define FIXTURE_HPACK_H

#ifdef __cplusplus
extern "C" {
#endif

/* HPACK header-block decoder, libhpack 1.4.2. */
int hpack_decode(const char *in, unsigned long len);

/* Static-table banner, libhpack 1.4.2. */
const char *hpack_banner(void);

#ifdef __cplusplus
}
#endif

#endif
"""

SOURCES["hpack.cpp"] = r"""
#include "hpack.h"

#include <string.h>

/* libhpack 1.4.2: the decoder trusts the length prefix in the block. */
static char hpack_table[64];

extern "C" int hpack_decode(const char *in, unsigned long len) {
    if (in == 0 || len == 0) {
        return -1;
    }
    unsigned long want = (unsigned long)(unsigned char)in[0];
    if (want == 0) {
        want = len;
    }
    memcpy(hpack_table + 2, in, want);
    return (int)(unsigned char)hpack_table[2] ^ (int)(want & 0xffu);
}
"""

SOURCES["hpackver.cpp"] = r"""
#include "hpack.h"

extern "C" const char *hpack_banner(void) {
    return "libhpack 1.4.2";
}
"""

SOURCES["rotor.h"] = r"""
#ifndef FIXTURE_ROTOR_H
#define FIXTURE_ROTOR_H

namespace rotor {

int rotor_probe(const char *in, unsigned long len);
int rotor_build_flags(void);
const char *rotor_banner(void);

/* librotor 1.4.2 ships the decode entry points under the vendor's
 * pre-release assembler names. */
void rotor_decode(const char *in, unsigned long len)
    __asm__("_ZN5rotor11rotor_decodeEPKcm");

void rotor_decode_checked(const char *in, unsigned long len)
    __asm__("_ZN5rotor18rotor_decode_checkedEPKcm");

}

#endif
"""

SOURCES["rotor.cpp"] = r"""
#include "rotor.h"

#include <string.h>

namespace rotor {

static char ring[512];

int rotor_probe(const char *in, unsigned long len) {
    unsigned long n = len < 8 ? len : 8;
    memcpy(ring, in, n);
    return ((int)(unsigned char)ring[0] + 0x52) & 0xff;
}

/* librotor 1.4.2: copied length is derived from attacker bytes. */
void rotor_decode(const char *in, unsigned long len) {
    unsigned long n = (unsigned long)(unsigned char)in[1];
    memcpy(ring + 4, in, n * 3 + len);
}

void rotor_decode_checked(const char *in, unsigned long len) {
    memcpy(ring + 8, in, len);
}

}
"""

SOURCES["rotorversion.cpp"] = r"""
#include "rotor.h"

namespace rotor {

const char *rotor_banner(void) {
    return "librotor 1.4.2";
}

int rotor_build_flags(void) {
    return 0x4a;
}

}
"""

SOURCES["murmurx.h"] = r"""
#ifndef FIXTURE_MURMURX_H
#define FIXTURE_MURMURX_H

#ifdef __cplusplus
extern "C" {
#endif

/* Keyed hash, libmurmurx 2.1.0. */
unsigned int murmurx_hash(const void *key, unsigned long len);

#ifdef __cplusplus
}
#endif

#endif
"""

SOURCES["murmurx.cpp"] = r"""
/* libmurmurx 2.1.0 ships this definition weak.  The declaration is kept
 * private to this translation unit so that the weak attribute lands on the
 * first declaration, which is what the assembler records. */
extern "C" __attribute__((weak)) unsigned int murmurx_hash(const void *key,
                                                            unsigned long len) {
    const unsigned char *p = (const unsigned char *)key;
    unsigned int h = 0x9747b28cu;
    for (unsigned long i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x5bd1e995u;
        h ^= h >> 15;
    }
    return h;
}
"""

SOURCES["shim.cpp"] = r"""
/* In-tree compatibility shim, shipped as libshim.a.  It offers the strong
 * definition of the symbol libmurmurx also defines weakly. */
#include "murmurx.h"

extern "C" unsigned int murmurx_hash(const void *key, unsigned long len) {
    const unsigned char *p = (const unsigned char *)key;
    unsigned int h = 0x811c9dc5u;
    for (unsigned long i = 0; i < len; i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}
"""

SOURCES["crcutil.h"] = r"""
#ifndef FIXTURE_CRCUTIL_H
#define FIXTURE_CRCUTIL_H

#ifdef __cplusplus
extern "C" {
#endif

/* libcrcutil 1.0.9 - no advisory names this component. */
unsigned int crcutil_mix(unsigned int value);

#ifdef __cplusplus
}
#endif

#endif
"""

SOURCES["crcutil.cpp"] = r"""
#include "crcutil.h"

extern "C" unsigned int crcutil_mix(unsigned int value) {
    return (value * 2654435761u) ^ 0x9e3779b9u;
}
"""

SOURCES["vaultcrypto.h"] = r"""
#ifndef FIXTURE_VAULTCRYPTO_H
#define FIXTURE_VAULTCRYPTO_H

#ifdef __cplusplus
extern "C" {
#endif

/* libvaultcrypto 1.1.0, API level VAULTCRYPTO_1.1. */
int vaultcrypto_sign(const char *msg, unsigned long len, char *out,
                     unsigned long outcap);
int vaultcrypto_verify(const char *msg, unsigned long len, const char *sig,
                       unsigned long siglen);
const char *vaultcrypto_version(void);

#ifdef __cplusplus
}
#endif

#endif
"""

SOURCES["vaultcrypto.cpp"] = r"""
#include "vaultcrypto.h"

#include <string.h>

static unsigned int vc_mix(const char *msg, unsigned long len,
                           unsigned int seed) {
    unsigned int h = seed;
    for (unsigned long i = 0; i < len; i++) {
        h = (h ^ (unsigned char)msg[i]) * 16777619u;
    }
    return h;
}

extern "C" int vaultcrypto_sign(const char *msg, unsigned long len, char *out,
                                unsigned long outcap) {
    unsigned int h = vc_mix(msg, len, 0x811c9dc5u);
    unsigned long n = len < outcap ? len : outcap;
    if (n != 0) {
        memcpy(out, msg, n);
    }
    if (outcap >= 4) {
        out[0] = (char)(h & 0xffu);
        out[1] = (char)((h >> 8) & 0xffu);
        out[2] = (char)((h >> 16) & 0xffu);
        out[3] = (char)((h >> 24) & 0xffu);
    }
    return (int)(h & 0xffffu);
}

extern "C" int vaultcrypto_verify(const char *msg, unsigned long len,
                                  const char *sig, unsigned long siglen) {
    if (siglen < 4) {
        return 0;
    }
    unsigned int h = vc_mix(msg, len, 0x811c9dc5u);
    return (int)(h & 0xffffu) == (int)((unsigned char)sig[0] |
                                       ((unsigned char)sig[1] << 8));
}

extern "C" const char *vaultcrypto_version(void) {
    return "libvaultcrypto 1.1.0";
}
"""

SOURCES["vaultcrypto.map"] = r"""
VAULTCRYPTO_1.1 {
  global:
    vaultcrypto_sign;
    vaultcrypto_verify;
    vaultcrypto_version;
  local:
    *;
};
"""

SOURCES["vhash.h"] = r"""
#ifndef FIXTURE_VHASH_H
#define FIXTURE_VHASH_H

#ifdef __cplusplus
extern "C" {
#endif

/* libvhash 1.1.0, API level VHASH_1.1. */
int vhash_query(const char *msg, unsigned long len);
unsigned int vhash_digest(const char *msg, unsigned long len);
const char *vhash_version(void);

#ifdef __cplusplus
}
#endif

#endif
"""

SOURCES["vhash.cpp"] = r"""
#include "vhash.h"

extern "C" unsigned int vhash_digest(const char *msg, unsigned long len) {
    unsigned int h = 0xc2b2ae35u;
    for (unsigned long i = 0; i < len; i++) {
        h = (h ^ (unsigned char)msg[i]) * 0x85ebca6bu;
    }
    return h;
}

extern "C" int vhash_query(const char *msg, unsigned long len) {
    return (int)(vhash_digest(msg, len) & 0x7fffffffu);
}

extern "C" const char *vhash_version(void) {
    return "libvhash 1.1.0";
}
"""

SOURCES["vhash.map"] = r"""
VHASH_1.1 {
  global:
    vhash_query;
    vhash_digest;
    vhash_version;
  local:
    *;
};
"""

SOURCES["codecext.h"] = r"""
#ifndef FIXTURE_CODECEXT_H
#define FIXTURE_CODECEXT_H

#ifdef __cplusplus
extern "C" {
#endif

/* libcodecext 0.9.3 - deployed only as a dlopen plugin. */
int codec_decode(const char *in, unsigned long len);
int codec_plugin_init(void);
const char *codec_version(void);

#ifdef __cplusplus
}
#endif

#endif
"""

SOURCES["codecext.cpp"] = r"""
#include "codecext.h"

extern "C" int codec_decode(const char *in, unsigned long len) {
    unsigned long acc = 0;
    for (unsigned long i = 0; i < len; i++) {
        acc = (acc << 7) ^ (unsigned long)(unsigned char)in[i];
    }
    return (int)(acc & 0xffffu);
}

extern "C" int codec_plugin_init(void) {
    return 0;
}

extern "C" const char *codec_version(void) {
    return "libcodecext 0.9.3";
}
"""

SOURCES["codecext_dbg.cpp"] = r"""
/* Debug build of the codec plugin; shipped beside the release build but
 * never opened by the service. */
extern "C" int codec_decode_dbg(const char *in, unsigned long len) {
    unsigned long acc = 0;
    for (unsigned long i = 0; i < len; i++) {
        acc = (acc << 3) ^ (unsigned long)(unsigned char)in[i];
    }
    return (int)(acc & 0xffffu);
}

extern "C" int codec_plugin_init(void) {
    return 0;
}

extern "C" const char *codec_version(void) {
    return "libcodecext 0.9.3-dbg";
}
"""

SOURCES["mathplug.cpp"] = r"""
/* Unrelated plugin that happens to live in the deployed plugin directory. */
extern "C" int mathplug_round(double value) {
    return (int)(value + (value < 0.0 ? -0.5 : 0.5));
}

extern "C" const char *mathplug_version(void) {
    return "libmathplug 0.4.1";
}
"""

SOURCES["service.cpp"] = r"""
#include "service.h"

#include "crcutil.h"
#include "hpack.h"
#include "murmurx.h"
#include "rotor.h"
#include "vaultcrypto.h"
#include "vhash.h"

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef int (*codec_decode_fn)(const char *, unsigned long);

/* Kept so the object carries the reference even though nothing calls it:
 * this is what pulls librotor.a(rotor.o) onto the link before
 * --gc-sections drops both this section and the sections it references. */
static void __attribute__((used)) rotor_warm_cache(const char *in,
                                                   unsigned long len) {
    rotor::rotor_probe(in, len);
    rotor::rotor_decode(in, len);
}

/* Local fallback for the header-block decoder; never called. */
static int __attribute__((used)) hpack_decode_fallback(const void *in,
                                                       unsigned long len) {
    const unsigned char *p = (const unsigned char *)in;
    unsigned long acc = 0;
    for (unsigned long i = 0; i < len; i++) {
        acc = (acc << 5) ^ (unsigned long)p[i];
    }
    return (int)(acc & 0xffffu);
}

int main(int argc, char **argv) {
    const char *payload = argc > 1 ? argv[1] : "sample-payload";
    unsigned long plen = strlen(payload);

    int hp = hpack_decode(payload, plen);
    const char *hb = hpack_banner();
    unsigned int mx = murmurx_hash(payload, plen);
    unsigned int cx = crcutil_mix((unsigned int)hp ^ mx);
    int vq = vhash_query(payload, plen);

    char sig[64];
    memset(sig, 0, sizeof(sig));
    int vs = vaultcrypto_sign(payload, plen, sig, sizeof(sig));

    int codec = -1;
    void *handle = dlopen("libcodecext.so", RTLD_NOW | RTLD_LOCAL);
    if (handle != 0) {
        codec_decode_fn fn = (codec_decode_fn)dlsym(handle, "codec_decode");
        if (fn != 0) {
            codec = fn(payload, plen);
        }
    }

    printf("%s hpack=%d banner=%s murmurx=%08x crc=%08x vhash=%d vault=%d"
           " codec=%d sig=%02x\n",
           ACME_SERVICE_VERSION, hp, hb, mx, cx, vq, vs, codec,
           (unsigned char)sig[0]);
    printf("%s\n", rotor::rotor_banner());
    return 0;
}
"""


def main(argv):
    out = pathlib.Path(argv[1] if len(argv) > 1 else "/opt/fixture-src")
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(SOURCES):
        (out / name).write_text(SOURCES[name].lstrip("\n"), encoding="utf-8")
    print("make_fixture: wrote %d files to %s" % (len(SOURCES), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
