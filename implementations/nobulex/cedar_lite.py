#!/usr/bin/env python3
"""
A small evaluator for the subset of Cedar that fixtures/policy/autoresearch-safe.cedar
uses. Not a Cedar implementation, and it says so where it matters.

It exists so this driver derives its decisions from the policy file rather than
from the fixtures' `expected_decision`. A driver that reads `expected_decision`
is copying the answer key, and the suite would report it conformant while
proving nothing. See issue #13, finding 6.

Supported subset, which is everything the policy uses:

    permit ( principal, action in [Action::"A", Action::"B"], resource );
    permit ( principal, action == Action::"A", resource ) when { <cond> };
    forbid ( principal, action == Action::"A", resource ) when { <cond> };

    <cond> := [ "v1", "v2" ].contains( context.<key> )
            | context.<key> == "v"

Cedar semantics applied here: an unmatched request is denied (default deny),
and a matching `forbid` overrides any matching `permit`.

WHAT THIS REFUSES, AND WHY THE REFUSAL IS THE POINT

`context.<key> in [ "v1", "v2" ]` is now rejected rather than evaluated. In
Cedar, `in` is the entity-hierarchy operator, and a string on its left is a
type error:

    type error: expected (entity of type `any_entity_type`), got string
    `in` is for checking the entity hierarchy; use `.contains()` to test set
    membership

This evaluator used to read `in` as list membership. The fixture policy was
written the same way, so an invalid policy was accepted here and produced the
intended decisions, and they matched `expected_decision` because the same hand
wrote both. cedar-wasm rejected that policy outright, and under protect-mcp's
fail-closed rule a policy error is a deny, which is why the reference denied
every Bash call while this driver allowed them. Reported by @tomjwxf on #12.

A permissive subset is worse than a missing one. It reports agreement with a
reference engine that is refusing its input, and the agreement is an artifact
rather than a measurement. So the subset must not accept what the reference
refuses: anything this file cannot evaluate the way Cedar would now raises
instead of guessing.
"""

import re
from typing import Any, Dict, List, Optional

_ACTION = re.compile(r'Action::"([^"]+)"')
# Set membership, the form Cedar actually accepts: the set is the receiver.
_CONTAINS = re.compile(r'\[([^\]]*)\]\s*\.\s*contains\s*\(\s*context\.(\w+)\s*\)',
                       re.S)
# The invalid form, matched only so it can be named and refused. A string on
# the left of `in` is a Cedar type error, and silently treating it as set
# membership is what let an invalid policy through.
_STRING_IN_LIST = re.compile(r'context\.(\w+)\s+in\s*\[([^\]]*)\]', re.S)
_EQ = re.compile(r'context\.(\w+)\s*==\s*"([^"]*)"')
_STRING = re.compile(r'"([^"]*)"')


class PolicyTypeError(ValueError):
    """The policy is not valid Cedar, so no decision is derived from it.

    Distinct from "this evaluator does not cover that syntax". Cedar would
    reject this input too, and reporting it as unsupported would imply the
    policy is fine and the evaluator is narrow, which is backwards.
    """


class Rule:
    def __init__(self, effect: str, actions: List[str], condition: Optional[dict]):
        self.effect = effect
        self.actions = actions
        self.condition = condition

    def matches(self, tool_name: str, context: Dict[str, Any]) -> bool:
        if self.actions and tool_name not in self.actions:
            return False
        if self.condition is None:
            return True
        got = context.get(self.condition["key"])
        return got in self.condition["values"]

    def __repr__(self):
        return f"<{self.effect} {self.actions} {self.condition}>"


def _strip_comments(text: str) -> str:
    return re.sub(r'//[^\n]*', '', text)


def parse(policy_text: str) -> List[Rule]:
    text = _strip_comments(policy_text)
    rules: List[Rule] = []
    for match in re.finditer(r'\b(permit|forbid)\b(.*?);', text, re.S):
        effect, body = match.group(1), match.group(2)

        # The head was scanned for Action:: and nothing else, so a constraint
        # on principal or resource was read as no constraint at all, and
        # Rule.matches treats no constraint as matching everything. A policy
        # reading `permit (principal == User::"alice", ...)` therefore returned
        # allow for every principal, which is the opposite of what it says.
        #
        # This is the rule the module docstring already states, applied to the
        # head rather than only to the guard. A subset that silently widens a
        # policy is worse than no subset: it reports agreement with a reference
        # engine while evaluating a different policy.
        for scope in ("principal", "resource"):
            if re.search(r'\b%s\s*(==|\bin\b|\bis\b)' % scope, body):
                raise PolicyTypeError(
                    "this policy constrains `%s`, and this evaluator reads "
                    "only the action from a rule head. Evaluating it would "
                    "drop the constraint and return a decision for principals "
                    "or resources the policy never permitted. Cedar enforces "
                    "it, so refusing is the only answer that agrees with the "
                    "reference engine." % scope)

        head, sep, guard = body.partition("when")

        # `unless { B }` is the negation of a `when` guard. Partitioning on
        # "when" left it sitting in the head, where nothing reads it, so the
        # clause vanished and its rule matched unconditionally.
        if "unless" in body:
            raise PolicyTypeError(
                "`unless` is not evaluated by this subset. It is the negation "
                "of a `when` guard, and leaving it unparsed makes the rule "
                "match unconditionally, which inverts what the policy says.")

        actions = _ACTION.findall(head)
        condition = None
        if guard.strip():
            contains = _CONTAINS.search(guard)
            bad_in = _STRING_IN_LIST.search(guard)
            equality = _EQ.search(guard)
            if contains:
                condition = {"key": contains.group(2), "op": "contains",
                             "values": _STRING.findall(contains.group(1))}
            elif bad_in:
                # Refused, not evaluated. Cedar rejects this and so must the
                # subset, or the subset clears policies the reference engine
                # will not run.
                raise PolicyTypeError(
                    "`context.%s in [ ... ]` is not valid Cedar: `in` is the "
                    "entity-hierarchy operator and a string on its left is a "
                    "type error. Use `[ ... ].contains(context.%s)` for set "
                    "membership. cedar-wasm rejects the policy as written, and "
                    "under a fail-closed engine a policy error is a deny, so "
                    "accepting it here would make this driver disagree with "
                    "every conforming one."
                    % (bad_in.group(1), bad_in.group(1)))
            elif equality:
                condition = {"key": equality.group(1), "op": "==",
                             "values": [equality.group(2)]}
            else:
                raise ValueError(
                    "unsupported `when` clause; this evaluator covers only the "
                    "subset the test-vector policy uses: " + guard.strip()[:80])
        rules.append(Rule(effect, actions, condition))
    if not rules:
        raise ValueError("no permit or forbid statements found in the policy")
    return rules


def evaluate(rules: List[Rule], tool_name: str, context: Dict[str, Any]) -> str:
    """Return "allow" or "deny". forbid wins; unmatched is denied."""
    for rule in rules:
        if rule.effect == "forbid" and rule.matches(tool_name, context):
            return "deny"
    for rule in rules:
        if rule.effect == "permit" and rule.matches(tool_name, context):
            return "allow"
    return "deny"
