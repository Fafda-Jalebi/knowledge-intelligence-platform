# Archival Description Guide

## 1 Authority Control

Authority control reconciles variant forms of a name to a single preferred heading so that a catalogue collocates all works by one author regardless of how the name was transcribed.

Example: "Twain, Mark" = "Clemens, Samuel Langhorne" = "Mark Twain"
- Preferred form established by national library (LCNAF, VIAF)
- Variants recorded as 4xx fields in MARC authority record
- Enables consistent retrieval across systems

## 2 Original Order

The principle of original order requires that records be kept in the arrangement imposed by their creator, because that arrangement is itself evidence of how the creator worked.

Violations of original order:
- Alphabetical rearrangement
- Chronological re-filing
- Subject-based classification imposed later

Respect des fonds: The fonds (the whole of the records of one creator) must not be intermixed with other fonds.

## 3 Description Standards

ISAD(G): General International Standard Archival Description
- Identity statement
- Context (archival history, immediate source)
- Content and structure
- Conditions of access and use
- Allied materials

DACS: Describing Archives: A Content Standard (US implementation of ISAD(G))

EAD: Encoded Archival Description (XML schema for finding aids)

## 4 Digital Preservation

Digital objects require:
- Fixity verification (checksums: SHA-256)
- Format migration planning (PRONOM registry)
- Metadata: PREMIS for preservation, METS for packaging
- Redundant storage (geographically distributed)
