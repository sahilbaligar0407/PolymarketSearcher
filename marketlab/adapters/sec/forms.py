"""Form 4 (ownership) XML parsing.

Form 4 reports an insider's change in beneficial ownership. The critical judgment call
this module makes explicit: **a routine, pre-scheduled transaction is not the same
evidence as a discretionary one.** An executive selling under a Rule 10b5-1(c) trading
plan adopted months earlier, or receiving a scheduled equity award, tells you nothing
about their current view of the stock - the sale was locked in before whatever news
prompted you to look. Only a genuinely discretionary open-market purchase or sale
(``code in {"P", "S"}`` without a 10b5-1 plan) is a signal worth treating as informative.

``is_routine`` is the single function every consumer of Form 4 data must call before
treating a transaction as bullish/bearish evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from xml.etree import ElementTree as ET

#: SEC Table I/II transaction codes -> human label. Not exhaustive of every obscure
#: code, but covers everything an insider-trading signal needs to reason about.
TRANSACTION_CODE_LABELS: dict[str, str] = {
    "P": "open_market_purchase",
    "S": "open_market_sale",
    "A": "grant_or_award",
    "D": "disposition_to_issuer",
    "F": "tax_withholding",
    "M": "option_exercise",
    "G": "gift",
    "C": "conversion",
    "X": "in_the_money_exercise",
    "V": "voluntary_report",
    "I": "discretionary_transaction",
    "J": "other",
    "K": "equity_swap",
    "U": "tender_of_shares",
    "W": "acquisition_or_disposition_by_will_or_trust",
    "Z": "deposit_or_withdrawal_from_voting_trust",
}

#: Transaction codes that are administrative/non-discretionary regardless of whether a
#: 10b5-1 plan is on file: grants, option exercises, tax withholding, gifts, conversions.
#: A discretionary open-market purchase (P) or sale (S) is only routine when a 10b5-1
#: plan flag or footnote confirms it was pre-scheduled.
_ROUTINE_CODES = {"A", "F", "M", "G", "C", "W", "Z"}

_TEN_B5_1_FOOTNOTE_HINT = "10b5-1"


@dataclass(frozen=True, slots=True)
class InsiderTransaction:
    """One row of a Form 4 - either a non-derivative or a derivative transaction table entry."""

    accession: str = ""
    cik: str = ""
    ticker: str = ""
    company: str = ""
    insider_name: str = ""
    #: officer / director / ten_percent_owner / other - a person can hold more than
    #: one; this is the highest-precedence role (director > officer > ten_percent_owner).
    insider_role: str = "other"
    officer_title: str = ""
    is_derivative: bool = False
    security_title: str = ""
    transaction_code: str = ""
    transaction_date: date | None = None
    shares: Decimal | None = None
    price_per_share: Decimal | None = None
    transaction_value: Decimal | None = None
    #: "A" (acquired) or "D" (disposed).
    acquired_disposed: str = ""
    shares_owned_after: Decimal | None = None
    ownership_nature: str = ""  # "D" direct / "I" indirect
    #: True when this transaction was executed under a disclosed Rule 10b5-1(c) plan.
    is_10b5_1: bool = False
    #: The filing's acceptance date, supplied by the caller (SEC adapter), used only to
    #: compute disclosure_lag_days here - the adapter is the source of published_time.
    filed_date: date | None = None
    #: Days between the transaction and its public disclosure. Part of the backtest -
    #: an insider trade is not "visible" to a strategy until this many days later.
    disclosure_lag_days: float | None = None

    @property
    def transaction_label(self) -> str:
        return TRANSACTION_CODE_LABELS.get(self.transaction_code, "unknown")


def is_routine(txn: InsiderTransaction) -> bool:
    """True if ``txn`` must NOT be treated as a discretionary bullish/bearish signal.

    - Any transaction flagged (document ``aff10b5One`` or a footnote mentioning
      "10b5-1") as executed under a Rule 10b5-1(c) trading plan is routine, whatever
      the transaction code - that is the entire point of such a plan: it was decided
      before the information environment that might otherwise explain it.
    - Grants/awards (A), option exercises (M), tax withholding (F), gifts (G),
      conversions (C), and a few other administrative codes are routine regardless of
      a 10b5-1 flag, since they are not a discretionary market view.
    - A bare open-market purchase (P) or sale (S) with no 10b5-1 flag is NOT routine -
      that is the signal insider-trading strategies actually want.
    """
    if txn.is_10b5_1:
        return True
    return txn.transaction_code in _ROUTINE_CODES


def _text(el: ET.Element | None) -> str:
    if el is None or el.text is None:
        return ""
    return el.text.strip()


def _value_text(parent: ET.Element | None, tag: str) -> str:
    """Most Form 4 fields are wrapped as ``<tag><value>...</value></tag>``.

    Searches descendants, not just direct children: ``transactionShares`` lives under
    ``transactionAmounts``, ``sharesOwnedFollowingTransaction`` under
    ``postTransactionAmounts``, etc. - callers shouldn't need to know the exact
    intermediate wrapper, and the tag names are unique within one transaction entry.
    """
    if parent is None:
        return ""
    node = parent.find(f".//{tag}")
    if node is None:
        return ""
    value_node = node.find("value")
    if value_node is not None:
        return _text(value_node)
    return _text(node)


def _to_decimal(s: str) -> Decimal | None:
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _to_date(s: str) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _bool(s: str) -> bool:
    return s.strip().lower() in {"true", "1", "yes"}


def _insider_role(relationship: ET.Element | None) -> tuple[str, str]:
    """Highest-precedence role plus officer title, from ``reportingOwnerRelationship``."""
    if relationship is None:
        return "other", ""
    is_director = _bool(_text(relationship.find("isDirector")))
    is_officer = _bool(_text(relationship.find("isOfficer")))
    is_ten_pct = _bool(_text(relationship.find("isTenPercentOwner")))
    title = _text(relationship.find("officerTitle"))
    if is_director:
        return "director", title
    if is_officer:
        return "officer", title
    if is_ten_pct:
        return "ten_percent_owner", title
    return "other", title


def _footnote_text_for(root: ET.Element, footnote_ids: set[str]) -> str:
    if not footnote_ids:
        return ""
    parts = []
    for fn in root.findall(".//footnotes/footnote"):
        fid = fn.get("id", "")
        if fid in footnote_ids and fn.text:
            parts.append(fn.text)
    return " ".join(parts)


def _footnote_ids_in(el: ET.Element) -> set[str]:
    return {fn.get("id", "") for fn in el.findall(".//footnoteId") if fn.get("id")}


def parse_form4(
    xml_text: str,
    *,
    accession: str = "",
    filed_date: date | datetime | None = None,
) -> list[InsiderTransaction]:
    """Parse a Form 4 ``ownershipDocument`` XML payload into insider transactions.

    Returns one :class:`InsiderTransaction` per non-derivative and derivative table
    entry (a single filing can report several transactions). Never raises on
    malformed/partial XML for fields it can't find - it returns as much as it could
    parse, since a Form 4 with a missing optional field is common and should not
    break ingestion of everything else in the filing.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    if isinstance(filed_date, datetime):
        filed_date = filed_date.date()

    issuer = root.find("issuer")
    cik = _text(issuer.find("issuerCik")) if issuer is not None else ""
    ticker = _text(issuer.find("issuerTradingSymbol")) if issuer is not None else ""
    company = _text(issuer.find("issuerName")) if issuer is not None else ""

    owner = root.find("reportingOwner")
    insider_name = ""
    role, officer_title = "other", ""
    if owner is not None:
        owner_id = owner.find("reportingOwnerId")
        insider_name = _text(owner_id.find("rptOwnerName")) if owner_id is not None else ""
        role, officer_title = _insider_role(owner.find("reportingOwnerRelationship"))

    #: Document-wide Rule 10b5-1(c) affirmation, added to the schema in 2023. Applies
    #: to whichever transaction(s) in the filing the footnotes/context indicate;
    #: conservatively, if this is set and there's exactly one transaction table entry
    #: (the overwhelmingly common case), it applies to that transaction.
    doc_10b5_1 = _bool(_text(root.find("aff10b5One")))

    transactions: list[InsiderTransaction] = []

    for table_tag, is_derivative in (("nonDerivativeTable", False), ("derivativeTable", True)):
        table = root.find(table_tag)
        if table is None:
            continue
        entry_tag = "derivativeTransaction" if is_derivative else "nonDerivativeTransaction"
        for entry in table.findall(entry_tag):
            coding = entry.find("transactionCoding")
            code = _value_text(coding, "transactionCode")

            shares = _to_decimal(_value_text(entry, "transactionShares"))
            price = _to_decimal(_value_text(entry, "transactionPricePerShare"))
            value = (shares * price) if (shares is not None and price is not None) else None

            footnote_ids = _footnote_ids_in(entry)
            footnote_text = _footnote_text_for(root, footnote_ids).lower()
            is_10b5_1 = doc_10b5_1 or (_TEN_B5_1_FOOTNOTE_HINT in footnote_text)

            txn_date = _to_date(_value_text(entry, "transactionDate"))
            lag_days: float | None = None
            if txn_date is not None and filed_date is not None:
                lag_days = float((filed_date - txn_date).days)

            transactions.append(
                InsiderTransaction(
                    accession=accession,
                    cik=cik,
                    ticker=ticker,
                    company=company,
                    insider_name=insider_name,
                    insider_role=role,
                    officer_title=officer_title,
                    is_derivative=is_derivative,
                    security_title=_value_text(entry, "securityTitle"),
                    transaction_code=code,
                    transaction_date=txn_date,
                    shares=shares,
                    price_per_share=price,
                    transaction_value=value,
                    acquired_disposed=_value_text(entry, "transactionAcquiredDisposedCode"),
                    shares_owned_after=_to_decimal(_value_text(entry, "sharesOwnedFollowingTransaction")),
                    ownership_nature=_value_text(entry, "directOrIndirectOwnership"),
                    is_10b5_1=is_10b5_1,
                    filed_date=filed_date,
                    disclosure_lag_days=lag_days,
                )
            )

    return transactions
