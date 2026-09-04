from __future__ import annotations

from datetime import date
from decimal import Decimal

from marketlab.adapters.sec.forms import is_routine, parse_form4

# A realistic Form 4 XML: a 10b5-1 plan sale by an officer, structurally identical to
# what EDGAR actually serves (nested transactionAmounts/postTransactionAmounts/
# ownershipNature wrappers, footnote-referenced 10b5-1 disclosure, aff10b5One flag).
_PLAN_SALE_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0609</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2026-09-01</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001780525</rptOwnerCik>
            <rptOwnerName>Newstead Jennifer</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>false</isDirector>
            <isOfficer>true</isOfficer>
            <isTenPercentOwner>false</isTenPercentOwner>
            <officerTitle>SVP, GC and Government Affairs</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <aff10b5One>true</aff10b5One>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value><footnoteId id="F1"/></securityTitle>
            <transactionDate><value>2026-09-01</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>S</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1439</value></transactionShares>
                <transactionPricePerShare><value>317.01</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>35790</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
    <footnotes>
        <footnote id="F1">This transaction was made pursuant to a Rule 10b5-1 trading plan adopted by the reporting person on May 5, 2026.</footnote>
    </footnotes>
    <ownerSignature>
        <signatureName>/s/ Sam Whittington, Attorney-in-Fact for Jennifer Newstead</signatureName>
        <signatureDate>2026-09-03</signatureDate>
    </ownerSignature>
</ownershipDocument>
"""

# An open-market purchase by a director - genuinely discretionary, not routine.
_OPEN_MARKET_PURCHASE_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4</documentType>
    <periodOfReport>2026-08-15</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0001111111</rptOwnerCik>
            <rptOwnerName>Board Member Jane</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>true</isDirector>
            <isOfficer>false</isOfficer>
            <isTenPercentOwner>false</isTenPercentOwner>
        </reportingOwnerRelationship>
    </reportingOwner>
    <aff10b5One>false</aff10b5One>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-08-15</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>P</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>500</value></transactionShares>
                <transactionPricePerShare><value>200.00</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>10500</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
    <ownerSignature>
        <signatureName>/s/ Jane Board Member</signatureName>
        <signatureDate>2026-08-17</signatureDate>
    </ownerSignature>
</ownershipDocument>
"""

# A scheduled equity award (code A) - routine regardless of any 10b5-1 flag.
_AWARD_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <documentType>4</documentType>
    <periodOfReport>2026-01-15</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0002222222</rptOwnerCik>
            <rptOwnerName>Exec Officer Sam</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>false</isDirector>
            <isOfficer>true</isOfficer>
            <isTenPercentOwner>false</isTenPercentOwner>
            <officerTitle>Chief Financial Officer</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <aff10b5One>false</aff10b5One>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Restricted Stock Unit</value></securityTitle>
            <transactionDate><value>2026-01-15</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>A</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>2000</value></transactionShares>
                <transactionPricePerShare><value>0</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>12000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
    <ownerSignature>
        <signatureName>/s/ Sam Exec Officer</signatureName>
        <signatureDate>2026-01-17</signatureDate>
    </ownerSignature>
</ownershipDocument>
"""


def test_parse_form4_plan_sale_realistic_fixture():
    txns = parse_form4(_PLAN_SALE_XML, accession="0001140361-26-035636", filed_date=date(2026, 9, 3))
    assert len(txns) == 1
    txn = txns[0]
    assert txn.cik == "0000320193"
    assert txn.ticker == "AAPL"
    assert txn.company == "Apple Inc."
    assert txn.insider_name == "Newstead Jennifer"
    assert txn.insider_role == "officer"
    assert txn.officer_title == "SVP, GC and Government Affairs"
    assert txn.transaction_code == "S"
    assert txn.shares == Decimal("1439")
    assert txn.price_per_share == Decimal("317.01")
    assert txn.transaction_value == Decimal("1439") * Decimal("317.01")
    assert txn.transaction_date == date(2026, 9, 1)
    assert txn.shares_owned_after == Decimal("35790")
    assert txn.is_10b5_1 is True
    assert txn.disclosure_lag_days == 2.0


def test_is_routine_true_for_10b5_1_plan_sale():
    txn = parse_form4(_PLAN_SALE_XML)[0]
    assert is_routine(txn) is True


def test_is_routine_true_for_award():
    txn = parse_form4(_AWARD_XML)[0]
    assert txn.transaction_code == "A"
    assert txn.is_10b5_1 is False
    assert is_routine(txn) is True


def test_is_routine_false_for_open_market_purchase():
    txn = parse_form4(_OPEN_MARKET_PURCHASE_XML)[0]
    assert txn.transaction_code == "P"
    assert txn.insider_role == "director"
    assert txn.is_10b5_1 is False
    assert is_routine(txn) is False


def test_disclosure_lag_days_computed_for_purchase():
    txn = parse_form4(_OPEN_MARKET_PURCHASE_XML, filed_date=date(2026, 8, 17))[0]
    assert txn.transaction_date == date(2026, 8, 15)
    assert txn.disclosure_lag_days == 2.0


def test_disclosure_lag_days_none_without_filed_date():
    txn = parse_form4(_OPEN_MARKET_PURCHASE_XML)[0]
    assert txn.disclosure_lag_days is None


def test_malformed_xml_returns_empty_list_not_raise():
    assert parse_form4("<not><valid") == []
    assert parse_form4("") == []


def test_transaction_label_property():
    txn = parse_form4(_AWARD_XML)[0]
    assert txn.transaction_label == "grant_or_award"
