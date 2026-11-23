# Field notes - certificate path

These are my notes on the certificate reader, written while I was chasing the
interop failures with the third-party signing tools.

* The reader walks tags naively. It does not build a schema; it reads a
  sequence of TLVs and recurses into whatever is constructed. As long as the
  tags line up it does not care how deeply things nest.

* It ignores the canonicality of lengths. Short form, long form, long form with
  leading zero octets, indefinite length - the reader takes all of them and
  never looks at the length octets again once it has the count.

* It ignores the canonicality of string construction. A string that has been
  split into segments and written as a constructed string is stitched back
  together and read as if it had been primitive all along.

* It accepts explicitly written DEFAULTs. When a component has a DEFAULT it
  does not mind the component being present and equal to the default; it reads
  the value it finds and keeps going.

* It accepts padded INTEGERs. Every INTEGER is reassembled from its content
  octets with the leading redundant octets left in place and the sign taken
  from the first of them, so sign-extended and zero-padded values read the
  same as their minimal encodings.

* It rejects a RelativeDistinguishedName whose SET OF children are not in
  ascending order. That is the one thing it will not take; everything else it
  either ignores or normalises.

I never did get to the same level of detail on the CMS side of the appliance -
that part is a linked library, not our code, and I was told not to touch it.
