"""GBNF grammar for schema-valid SLM JSON output.

Constrained decoding guarantees the shape required by ``SLMAnalysis`` so the
production path never falls back to UNKNOWN because of malformed JSON. Content
accuracy is still the model's responsibility: the grammar only fixes the
structure, enums, and value ranges.

The grammar deliberately avoids ``{n,m}`` repetition after a rule reference:
llama.cpp 0.3.x can mis-parse it once the pandas/xgboost stack is loaded. The
model's outputs are already within the schema's length limits; pydantic enforces
them.
"""
from __future__ import annotations

SLM_JSON_GRAMMAR = r'''
root ::= "{" ws "\"risk_level\"" ws ":" ws risk ws "," ws "\"likely_condition\"" ws ":" ws string ws "," ws "\"confidence\"" ws ":" ws confidence ws "," ws "\"evidence\"" ws ":" ws evidence ws "," ws "\"recommended_action\"" ws ":" ws string ws "," ws "\"urgency\"" ws ":" ws urgency ws "}"
risk ::= "\"LOW\"" | "\"MEDIUM\"" | "\"HIGH\"" | "\"CRITICAL\"" | "\"UNKNOWN\""
urgency ::= "\"ROUTINE\"" | "\"PRIORITY\"" | "\"IMMEDIATE\"" | "\"UNKNOWN\""
confidence ::= "0" ("." [0-9]+)? | "1" ("." "0"+)?
ws ::= [ \t\n]*
string ::= "\"" char+ "\""
char ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
evidence ::= "[" ws (string (ws "," ws string)*) ws "]"
'''
