# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Atomic text operations for the 3-role reflection architecture.

Defines the :class:`EditTool` enum, the typed arguments each tool takes, and
:func:`apply_edit`, which runs one tool on a region of text and reports the
INSERT/DELETE operations it decomposed into.

Edit-tool basis
---------------
Every edit reduces to the two-operation minimal basis ``{INSERT_TEXT,
DELETE_TEXT}`` (as in the Levenshtein Transformer [1]_, operational
transformation [2]_, and CRDT [3]_ literatures). The optional broad set promotes
the two highest-frequency compositions, ``REPLACE_TEXT`` and ``MOVE_TEXT`` —
exactly the four operations that tree-diffing (GumTree [4]_ ``{add, delete,
updateValue, move}``) and string-edit theory (Damerau-Levenshtein [5]_ [6]_
``{insert, delete, substitute, transpose}``) independently converge on. Which
basis is offered is an ablation axis (:data:`EDIT_TOOL_SETS`).

Each reference below names the section that states the claim and links straight
to that page.

.. [1] Gu, Wang & Zhao. "Levenshtein Transformer." NeurIPS 2019.
   §2.2 "Actions: Deletion & Insertion" — "the two basic actions - deletion and
   insertion". https://arxiv.org/pdf/1905.11006#page=2
.. [2] Ellis & Gibbs. "Concurrency Control in Groupware Systems." SIGMOD 1989.
   §3 "The Model" — a text site object's operator set is
   ``O = {O1 = insert[X; P], O2 = delete[P]}`` (§4.1 "Transformation Matrix"
   then builds all conflict resolution on that pair).
   https://www.lri.fr/~mbl/ENS/CSCW/2012/papers/Ellis-SIGMOD89.pdf#page=4
   (DOI: https://doi.org/10.1145/67544.66963)
.. [3] Oster, Urso, Molli & Imine. "Data Consistency for P2P Collaborative
   Editing" (WOOT). CSCW 2006. §2 "WOOT Approach" — "Every editing action of a
   linear structure can be expressed in terms of the two following primitive
   operations: ins(...) [and] del(e)".
   https://hal.inria.fr/inria-00108523/document#page=3
   (DOI: https://doi.org/10.1145/1180875.1180916)
.. [4] Falleri, Morandat, Blanc, Martinez & Monperrus. "Fine-grained and
   Accurate Source Code Differencing." ASE 2014. §2 "AST Differencing" — edit
   actions ``updateValue``, ``add``, ``delete``, ``move``.
   https://hal.science/hal-01054552/document#page=3
   (DOI: https://doi.org/10.1145/2642937.2642982)
.. [5] Levenshtein. "Binary Codes Capable of Correcting Deletions, Insertions,
   and Reversals." Soviet Physics Doklady 10(8):707-710, 1966. p. 707: opening
   paragraph defines deletions, insertions and reversals; §1 "Codes Capable of
   Correcting Deletions and Insertions" defines the distance rho(x, y) as the
   smallest number of deletions and insertions.
   https://nymity.ch/sybilhunting/pdf/Levenshtein1966a.pdf#page=1
.. [6] Damerau. "A Technique for Computer Detection and Correction of Spelling
   Errors." Communications of the ACM 7(3):171-176, 1964. Abstract (p. 171): a
   single error is "a wrong, missing or extra letter or a single transposition".
   https://doi.org/10.1145/363958.363994
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class EditApplicationError(ValueError):
    """Raised when an :class:`EditTool` cannot be applied to a region.

    :func:`apply_edit` raises it when a required argument is empty or when the
    edit's ``target``/``anchor`` does not occur in the region text, so a
    Proposer can report the failure back to the LM and retry with a valid edit
    instead of silently leaving the region unchanged.
    """


class EditTool(str, Enum):
    """An atomic text operation that may modify a selected :class:`~gepa.strategies.document_template.EditTarget`.

    ``INSERT_TEXT`` and ``DELETE_TEXT`` form the minimal complete basis; every
    other operation decomposes into them (recorded in the atomic-op log so
    step-8 logging shows the decomposition). ``REPLACE_TEXT`` and ``MOVE_TEXT``
    are the two promoted high-frequency compositions of the broad set.
    """

    INSERT_TEXT = "INSERT_TEXT"
    DELETE_TEXT = "DELETE_TEXT"
    REPLACE_TEXT = "REPLACE_TEXT"
    MOVE_TEXT = "MOVE_TEXT"


# The EditTool ablation axis: minimal 2-op basis vs. broad 4-op set.
EDIT_TOOL_SETS: dict[str, list[EditTool]] = {
    "minimal": [EditTool.INSERT_TEXT, EditTool.DELETE_TEXT],
    "broad": [EditTool.INSERT_TEXT, EditTool.DELETE_TEXT, EditTool.REPLACE_TEXT, EditTool.MOVE_TEXT],
}


# Which side of an anchor substring an edit lands on.
Placement = Literal["before", "after"]


@dataclass(frozen=True)
class InsertTextArgs:
    """Arguments for :attr:`EditTool.INSERT_TEXT`.

    Args:
        text: The new text to insert, applied verbatim (whitespace preserved).
        anchor: Existing substring of the region to insert next to. Empty means
            "no anchor": the text is appended at the end of the region.
        where: Which side of ``anchor`` the text goes on.
    """

    text: str
    anchor: str = ""
    where: Placement = "after"


@dataclass(frozen=True)
class DeleteTextArgs:
    """Arguments for :attr:`EditTool.DELETE_TEXT`.

    Args:
        target: Existing substring of the region to remove.
    """

    target: str


@dataclass(frozen=True)
class ReplaceTextArgs:
    """Arguments for :attr:`EditTool.REPLACE_TEXT`.

    Args:
        target: Existing substring of the region to replace.
        text: What to put in its place; may be empty, which is a plain delete.
    """

    target: str
    text: str


@dataclass(frozen=True)
class MoveTextArgs:
    """Arguments for :attr:`EditTool.MOVE_TEXT`.

    Args:
        target: Existing substring of the region to relocate.
        anchor: Existing substring marking the destination; must still be
            present once ``target`` has been cut out.
        where: Which side of ``anchor`` the moved text lands on.
    """

    target: str
    anchor: str
    where: Placement = "after"


EditArgs = InsertTextArgs | DeleteTextArgs | ReplaceTextArgs | MoveTextArgs


def _insert_text(region: str, args: InsertTextArgs) -> tuple[str, list[str]]:
    """Insert new text into ``region``, either next to an anchor or at the very end.

    If ``args.anchor`` is empty, ``args.text`` is appended to the end of
    ``region``. Otherwise the first occurrence of ``args.anchor`` is located and
    ``args.text`` is placed immediately before it (``where == "before"``) or
    immediately after it (``where == "after"``). No other part of ``region``
    changes.

    Args:
        region: Text of the region being edited.
        args: The text to insert, the anchor to place it next to, and the side.

    Returns:
        The edited region, and a one-entry log naming the INSERT that ran
        (e.g. ``["INSERT ' big' after 'hello'"]``, or ``"... at end"``).

    Raises:
        EditApplicationError: ``args.text`` is empty, or ``args.anchor`` is
            non-empty but does not occur in ``region``.
    """
    if not args.text:
        raise EditApplicationError("INSERT_TEXT requires non-empty 'text'.")
    if not args.anchor:
        return region + args.text, [f"INSERT {args.text!r} at end"]
    idx = region.find(args.anchor)
    if idx < 0:
        raise EditApplicationError(f"INSERT_TEXT anchor not found: {args.anchor!r}")
    if args.where == "before":
        new_region = region[:idx] + args.text + region[idx:]
    else:
        end = idx + len(args.anchor)
        new_region = region[:end] + args.text + region[end:]
    return new_region, [f"INSERT {args.text!r} {args.where} {args.anchor!r}"]


def _delete_text(region: str, args: DeleteTextArgs) -> tuple[str, list[str]]:
    """Remove one occurrence of a substring from ``region``.

    Only the first (leftmost) occurrence of ``args.target`` is removed; later
    occurrences are left untouched.

    Args:
        region: Text of the region being edited.
        args: The substring to remove.

    Returns:
        The edited region, and a one-entry log naming the DELETE that ran
        (e.g. ``["DELETE ' world'"]``).

    Raises:
        EditApplicationError: ``args.target`` is empty or does not occur in
            ``region``.
    """
    if not args.target:
        raise EditApplicationError("DELETE_TEXT requires non-empty 'target'.")
    if args.target not in region:
        raise EditApplicationError(f"DELETE_TEXT target not found: {args.target!r}")
    return region.replace(args.target, "", 1), [f"DELETE {args.target!r}"]


def _replace_text(region: str, args: ReplaceTextArgs) -> tuple[str, list[str]]:
    """Swap one occurrence of a substring in ``region`` for new text.

    The first (leftmost) occurrence of ``args.target`` is removed and
    ``args.text`` is put in the same position, so surrounding text is unchanged.
    An empty ``args.text`` simply deletes the target.

    Args:
        region: Text of the region being edited.
        args: The substring to replace and its replacement.

    Returns:
        The edited region, and a two-entry log showing the operation as its
        atomic parts, ``["DELETE <target>", "INSERT <text>"]``.

    Raises:
        EditApplicationError: ``args.target`` is empty or does not occur in
            ``region``.
    """
    if not args.target:
        raise EditApplicationError("REPLACE_TEXT requires non-empty 'target'.")
    if args.target not in region:
        raise EditApplicationError(f"REPLACE_TEXT target not found: {args.target!r}")
    return region.replace(args.target, args.text, 1), [f"DELETE {args.target!r}", f"INSERT {args.text!r}"]


def _move_text(region: str, args: MoveTextArgs) -> tuple[str, list[str]]:
    """Cut a substring out of ``region`` and paste it next to an anchor.

    The first occurrence of ``args.target`` is removed, and then, in the text
    that remains, the first occurrence of ``args.anchor`` is located and the cut
    text is inserted immediately before or after it per ``args.where``. Because
    the anchor is searched for *after* the cut, an anchor that overlapped the
    target no longer exists and the move fails.

    Args:
        region: Text of the region being edited.
        args: The substring to move, the anchor marking its destination, and
            the side of the anchor to land on.

    Returns:
        The edited region, and a two-entry log showing the operation as its
        atomic parts, ``["DELETE <target>", "INSERT (moved) <target> ..."]``.

    Raises:
        EditApplicationError: ``args.target`` or ``args.anchor`` is empty,
            ``args.target`` does not occur in ``region``, or ``args.anchor``
            is not found once ``args.target`` has been removed.
    """
    if not args.target:
        raise EditApplicationError("MOVE_TEXT requires non-empty 'target'.")
    if args.target not in region:
        raise EditApplicationError(f"MOVE_TEXT target not found: {args.target!r}")
    if not args.anchor:
        raise EditApplicationError("MOVE_TEXT requires a non-empty 'anchor' destination.")
    removed = region.replace(args.target, "", 1)
    idx = removed.find(args.anchor)
    if idx < 0:
        raise EditApplicationError(f"MOVE_TEXT anchor not found after removal: {args.anchor!r}")
    if args.where == "before":
        new_region = removed[:idx] + args.target + removed[idx:]
    else:
        end = idx + len(args.anchor)
        new_region = removed[:end] + args.target + removed[end:]
    return new_region, [f"DELETE {args.target!r}", f"INSERT (moved) {args.target!r} {args.where} {args.anchor!r}"]


def apply_edit(region: str, args: EditArgs) -> tuple[str, list[str]]:
    """Run one edit on ``region`` and report the atomic operations it took.

    The type of ``args`` selects the operation: :class:`InsertTextArgs` inserts,
    :class:`DeleteTextArgs` deletes, :class:`ReplaceTextArgs` replaces, and
    :class:`MoveTextArgs` moves. REPLACE and MOVE are compositions of the
    INSERT/DELETE basis, and the returned log spells that decomposition out so
    the executed edit can be recorded exactly.

    Args:
        region: Text of the region being edited.
        args: The typed arguments of the edit to run.

    Returns:
        The edited region, and the log of atomic INSERT/DELETE operations that
        produced it (one entry for INSERT/DELETE, two for REPLACE/MOVE).

    Raises:
        EditApplicationError: A required argument is empty, or the edit's
            anchor/target does not occur in ``region``.
        TypeError: ``args`` is not one of the :data:`EditArgs` types.
    """
    if isinstance(args, InsertTextArgs):
        return _insert_text(region, args)
    if isinstance(args, DeleteTextArgs):
        return _delete_text(region, args)
    if isinstance(args, ReplaceTextArgs):
        return _replace_text(region, args)
    if isinstance(args, MoveTextArgs):
        return _move_text(region, args)
    raise TypeError(f"Unsupported edit args: {type(args).__name__}")
