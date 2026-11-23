# Loss contract

This document is normative. It fixes the scalar training loss of one record
and the per-position terms that make it up. Where an implementation of the
loss disagrees with a clause below, the implementation is wrong; the clause
identifiers are used verbatim when a loss is adjudicated, and the clause order
below is the order that adjudication walks.

A record carries

  * `logits`, an `N x V` array of unnormalised scores; row `i` holds the
    scores of position `i`;
  * `labels`, `N` integers.  A position whose label is `-100` (`ignore_index`)
    is *ignored*; every other position is *supervised*;
  * `label_smoothing`, written `eps` below, with `0 <= eps < 1`;
  * `reduction`, either `"sum"` or `"mean"`;
  * `count_policy`, an opaque registry id.

Write `z[i] = logits[i]`, `lse[i] = logsumexp(z[i])`, and
`logp[i][j] = z[i][j] - lse[i]`, so that `p[i] = softmax(z[i])`.

## C1 - Objective

The loss is the reduction (C4) of the per-position terms (C2, C3) formed over
the record's *carrier* `L`, a subset of the positions `{0, ..., N-1}`.  `L` is
the subset that the record's `count_policy` designates.  A position outside
`L` contributes nothing at all: it forms no C2 term and no C3 term, and it
does not enter the C4 divisor.  This document does not enumerate what any
`count_policy` designates.

## C2 - Log-likelihood term

Every supervised position `i` of `L` forms

    h[i] = -(1 - eps) * logp[i][labels[i]]
         =  (1 - eps) * (lse[i] - z[i][labels[i]]).

An ignored position forms no `h[i]`.

## C3 - Label-smoothing term

Every position `i` of `L`, supervised or ignored, forms

    u[i] = -eps * mean_j logp[i][j]
         =  (eps / V) * (lse[i] - mean_j z[i][j]).

The per-position term of position `i` is `t[i] = h[i] + u[i]`, and the
numerator of the loss is `T = sum over i in L of t[i]`.  An ignored position
of a non-empty carrier therefore still carries a term, `t[i] = u[i]`, which is
non-zero whenever `eps > 0`.

## C4 - Reduction and divisor

Let `M = |L|`.

  * `reduction = "sum"`: the loss is `T`, so the divisor is `1`.
  * `reduction = "mean"`: the loss is `T / M`, so the divisor is `M`.
  * For an empty carrier (`M = 0`) the loss is `0.0` and the divisor is `0`
    under `"mean"` and `1` under `"sum"`.  The mean of an empty carrier is
    *defined* to be `0.0`: it is not the quotient `0 / 0`, and that quotient
    is not evaluated.  An implementation that evaluates it anyway, or whose
    effective divisor for the record is anything else, contradicts this
    clause.

## C5 - Empty carrier

A record whose carrier is empty is well formed.  Its loss is the finite `0.0`
fixed by C4, and its derivative with respect to every logit is exactly zero.
