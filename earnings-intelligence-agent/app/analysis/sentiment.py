"""Finance-domain sentiment analyzer for SEC filing text.

Architecture:
    TF-IDF vectorizer (unigrams + bigrams, sublinear TF scaling)
    → Logistic Regression (multinomial softmax, balanced class weights)

Training data is synthesized from a curated corpus of finance-domain sentences
labeled as positive, neutral, or negative. No external dataset is required —
call ``SentimentAnalyzer().fit()`` to train on the internal corpus, or pass
your own ``(texts, labels)`` to override it.

Score semantics:
    score = P(positive) - P(negative)  →  range [-1.0, 1.0]
    label boundaries: score > 0.15 → "positive", < -0.15 → "negative", else "neutral"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle as sk_shuffle

# ── Constants ─────────────────────────────────────────────────────────────────

_LABEL_NEG = "negative"
_LABEL_NEU = "neutral"
_LABEL_POS = "positive"

# |score| below this boundary is classified as neutral
_SCORE_THRESHOLD = 0.15

# Long texts are split into chunks of this many words before scoring, then averaged
_CHUNK_WORDS = 150


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SentimentResult:
    """Sentiment analysis result for a single text.

    Attributes:
        score:      Compound score in [-1.0, 1.0]. Positive values indicate
                    bullish/optimistic language; negative values indicate
                    bearish/risk-heavy language.
        label:      One of "positive", "neutral", or "negative".
        confidence: Highest class probability from the softmax output [0.0, 1.0].
    """

    score: float
    label: str
    confidence: float


# ── Training corpus ───────────────────────────────────────────────────────────

# Each list contains finance-domain sentences representative of that class.
# Sentences are written to reflect real SEC 10-K / 10-Q / earnings release prose.

_POSITIVE_SENTENCES: list[str] = [
    # Revenue and earnings beats
    "Net revenue increased 18 percent year-over-year, driven by strong demand across all product lines.",
    "The company exceeded analyst consensus estimates for both revenue and earnings per share.",
    "Earnings per share of $4.32 beat the consensus estimate of $3.98 by approximately 8 percent.",
    "Quarterly revenue reached a record $89.5 billion, surpassing all prior quarterly records.",
    "Net income grew 24 percent year-over-year to $21.7 billion, reflecting strong operating leverage.",
    "Gross profit increased to $38.4 billion, with gross margin expanding 180 basis points.",
    "The company delivered its strongest quarterly results in its 40-year operating history.",
    "Revenue beat expectations by a wide margin, driven by exceptional performance in services.",
    "Operating income increased significantly, reflecting favorable mix shift and strong pricing power.",
    "Full-year earnings per share grew 30 percent, driven by revenue growth and share repurchases.",
    "Revenue growth accelerated to 20 percent in the final quarter, exceeding management guidance.",
    "The company reported record quarterly earnings, driven by robust demand in all geographies.",
    "Comparable sales grew 9 percent, significantly exceeding analyst expectations of 4 percent.",
    "Adjusted operating income reached an all-time high, reflecting disciplined cost management.",
    "The company delivered double-digit revenue growth for the eighth consecutive quarter.",
    # Margin and efficiency
    "Operating margins expanded by 200 basis points, reflecting improved operational efficiency.",
    "Gross margins reached an all-time high of 47.3 percent, demonstrating exceptional pricing strength.",
    "Cost optimization initiatives resulted in $2.1 billion in annualized savings ahead of schedule.",
    "Return on equity improved to 162 percent, reflecting outstanding capital allocation efficiency.",
    "The company achieved record operating leverage as fixed costs were spread over higher revenue.",
    "Productivity improvements drove meaningful cost reductions across all major business functions.",
    "Supply chain optimization efforts resulted in significant gross margin improvement year-over-year.",
    "Favorable product mix and pricing discipline drove gross margin to its highest level in a decade.",
    "Operational excellence initiatives contributed $1.4 billion in cost savings during the fiscal year.",
    "Expense discipline and scale benefits drove strong incremental operating margin of 45 percent.",
    # Cash flow and balance sheet
    "Free cash flow generation reached a record $12.3 billion for the fiscal year.",
    "The company returned $28 billion to shareholders through dividends and share repurchases.",
    "Cash and cash equivalents increased to $67 billion, providing substantial financial flexibility.",
    "The strong balance sheet supports continued investment in growth and shareholder return programs.",
    "Operating cash flow grew 20 percent year-over-year, enabling accelerated capital return programs.",
    "The company generated $15 billion in free cash flow, exceeding management's initial guidance.",
    "Net cash position strengthened to $38 billion, providing ample capacity for strategic investments.",
    "The company's robust cash generation funded a 10 percent dividend increase and $90 billion buyback.",
    "Debt-to-EBITDA ratio improved to 1.2 times, reflecting strong earnings growth and debt repayment.",
    "The company ended the quarter with $52 billion in cash and short-term investments.",
    # Guidance and outlook
    "Management raised full-year guidance, citing continued momentum in core markets.",
    "The company provided better-than-expected guidance for the upcoming fiscal year.",
    "Management expressed high confidence in achieving sustained double-digit revenue growth.",
    "Forward guidance exceeded analyst expectations across all key financial metrics.",
    "The company raised its quarterly dividend by 10 percent, reflecting confidence in earnings growth.",
    "Management issued upward guidance revisions for revenue, gross margin, and earnings per share.",
    "The company's strong order backlog supports management's optimistic outlook for the coming year.",
    "Guidance for the next fiscal year implies accelerating revenue growth and margin expansion.",
    "Management's outlook called for record revenue and earnings in the upcoming fiscal year.",
    "The company reaffirmed its long-term financial targets, citing strong competitive positioning.",
    # Market share and growth
    "Market share gains accelerated in the premium segment, reinforcing durable competitive advantages.",
    "The company expanded its customer base by 15 percent, driven by innovative new product launches.",
    "International revenue grew 22 percent year-over-year, driven by rapid emerging market expansion.",
    "The new product line achieved exceptional adoption rates, exceeding all internal projections.",
    "Strategic acquisitions strengthened the company's position in high-growth adjacent markets.",
    "Customer retention rates reached a record high, reflecting exceptional product quality and service.",
    "New contract wins totaled $8.2 billion during the quarter, an increase of 35 percent.",
    "The company's innovative products continued to drive strong pricing power across key markets.",
    "Revenue per customer increased 12 percent, reflecting successful upselling and cross-sell execution.",
    "The services segment grew 14 percent, diversifying revenue and improving earnings quality.",
    "The company outperformed peers across all key financial metrics for the third consecutive year.",
    "The company successfully expanded into new geographies, adding significant addressable market.",
    "The acquisition was immediately accretive to earnings and free cash flow per share.",
    "Backlog increased to a record $52 billion, providing exceptional near-term revenue visibility.",
    "Customer satisfaction scores reached an all-time high, supporting continued pricing power.",
    "The company's robust new product pipeline supports a positive multi-year growth outlook.",
    "Subscription revenue grew 38 percent, reflecting strong customer demand and low churn rates.",
    "The company gained significant market share in its largest segment for the fifth consecutive year.",
    "Organic revenue growth of 11 percent reflected strong underlying demand across all end markets.",
    "The company's strategic investments in innovation are generating attractive returns ahead of plan.",
    "Brand strength and loyal customers provide durable competitive advantages and pricing resilience.",
    "The company's platform strategy is creating powerful network effects that accelerate growth.",
    "Record new customer additions in the quarter reflect the strength of the company's value proposition.",
    "The company's digital transformation initiatives are driving productivity gains and revenue growth.",
    "Management's disciplined capital allocation has consistently created shareholder value.",
    "The company's differentiated product portfolio is winning new customers at an accelerating pace.",
]

_NEGATIVE_SENTENCES: list[str] = [
    # Impairments and write-downs
    "The company recorded a goodwill impairment charge of $1.2 billion related to its media segment.",
    "Asset impairment charges of $3.4 billion were recognized due to declining business prospects.",
    "The company wrote off $892 million in capitalized software costs following project cancellations.",
    "Restructuring and impairment charges totaled $2.1 billion, significantly impacting reported earnings.",
    "The company recognized a $4.7 billion write-down on acquired intangible assets.",
    "Impairment of long-lived assets resulted in a non-cash charge of $678 million for the period.",
    "The company recorded inventory write-downs of $340 million due to obsolescence and pricing pressure.",
    "Investment impairments of $1.1 billion reflected the deterioration in equity investee performance.",
    "The company recognized impairment of right-of-use assets related to vacated leased facilities.",
    "Deferred tax asset valuation allowances were recorded due to uncertainty about future profitability.",
    # Revenue and earnings misses
    "Revenue declined 12 percent year-over-year due to unfavorable macroeconomic conditions.",
    "Quarterly earnings per share of $1.23 missed the consensus estimate by 18 percent.",
    "The company reported a net loss of $1.8 billion for the fiscal year.",
    "Revenue fell short of expectations due to weaker than anticipated consumer demand.",
    "Operating income declined 34 percent as a result of intensified competition and pricing pressure.",
    "The company missed revenue guidance by $400 million, citing unexpected demand deterioration.",
    "Net income declined 45 percent year-over-year, driven by severe margin compression.",
    "Gross margin contracted 320 basis points due to unfavorable product mix and rising input costs.",
    "The company reported its third consecutive quarterly loss, raising concerns about long-term viability.",
    "Revenue from the core segment declined 19 percent, reflecting accelerating market share losses.",
    "Sales volumes decreased significantly due to reduced consumer spending and macro headwinds.",
    "The company's operating loss widened to $2.3 billion, reflecting continued investment overhang.",
    "Revenue declined across all three business segments, indicating broad-based demand deterioration.",
    "Earnings missed expectations for the fourth consecutive quarter, eroding investor confidence.",
    "The company's adjusted earnings per share declined 28 percent, well below guidance.",
    # Litigation and regulatory risk
    "The company faces significant litigation risk arising from multiple pending lawsuits.",
    "Regulatory investigations by the SEC and DOJ pose substantial risks to operations and reputation.",
    "The company is subject to antitrust investigations in multiple jurisdictions worldwide.",
    "Pending class action lawsuits could result in material financial penalties and reputational damage.",
    "The company received a $2.3 billion regulatory fine related to anti-competitive pricing practices.",
    "Environmental litigation could result in significant remediation costs and operational disruptions.",
    "The company faces substantial legal exposure from product liability claims across multiple markets.",
    "Regulatory compliance failures resulted in significant fines and mandatory operational changes.",
    "Legal settlements totaling $890 million adversely impacted financial results for the period.",
    "The company disclosed an ongoing criminal investigation that could result in severe penalties.",
    "Adverse court rulings could materially impair the company's ability to sell key products.",
    "The company accrued $1.5 billion for probable losses related to outstanding legal proceedings.",
    "Government investigations into the company's accounting practices created material uncertainty.",
    "Privacy regulatory actions exposed the company to significant financial and reputational risk.",
    "The company faces potential debarment from government contracts due to regulatory violations.",
    # Restructuring
    "The company announced a restructuring plan affecting approximately 8,000 employees globally.",
    "Workforce reduction charges of $400 million were incurred in connection with the restructuring.",
    "The company is restructuring its operations in response to dramatically deteriorating conditions.",
    "Significant restructuring costs are expected to continue through the next two fiscal years.",
    "The company eliminated 12 percent of its global workforce as part of an aggressive cost reduction.",
    "Factory closures and headcount reductions will result in $600 million in restructuring charges.",
    "The company accelerated its restructuring program following worse-than-expected operating results.",
    "Severance and facility exit costs of $280 million were recognized in connection with the layoffs.",
    "The company is exiting several underperforming businesses as part of its portfolio restructuring.",
    "Ongoing restructuring activities are expected to create continued operational disruption.",
    # Guidance cuts and cautious outlook
    "Management lowered full-year guidance, citing deteriorating demand and supply chain disruptions.",
    "The company withdrew its annual guidance entirely due to significant macroeconomic uncertainty.",
    "Forward guidance missed analyst expectations by a wide margin across all reported metrics.",
    "Management warned of continued headwinds expected to pressure results in coming quarters.",
    "The company issued profit warnings, reflecting a significant deterioration in the business outlook.",
    "Guidance was reduced for the third consecutive quarter, reflecting persistent operational challenges.",
    "Management expressed caution about near-term prospects, citing rising costs and weaker demand.",
    "The company's revised guidance implies a sharp sequential decline in both revenue and margins.",
    "Management cautioned that macro uncertainty makes reliable guidance impossible at this time.",
    "The company lowered its revenue outlook by $2 billion, citing slower-than-expected recovery.",
    # Debt and liquidity
    "The company's elevated debt levels and declining cash flows raise serious liquidity concerns.",
    "Covenant violations on the company's credit facility triggered immediate repayment requirements.",
    "The company's ability to continue as a going concern is subject to significant uncertainty.",
    "Debt refinancing risks could adversely impact financial flexibility and strategic optionality.",
    "The company breached its debt covenants and is in active negotiations with lenders.",
    "Available liquidity declined sharply, limiting the company's ability to invest in its business.",
    "Rising interest expense on variable rate debt is expected to further compress earnings.",
    "The company's leverage ratio increased to 5.8 times, well above its stated long-term target.",
    "Constrained access to capital markets could impair the company's ability to fund operations.",
    "The company drew down its full revolving credit facility, signaling acute liquidity pressure.",
    # Market and competition
    "Intense competition has eroded the company's market share and pricing power in key markets.",
    "The loss of a major customer represents a significant risk to near-term revenue stability.",
    "Adverse currency movements adversely impacted reported results by $1.2 billion.",
    "Market share losses in the core business accelerated amid intensifying competitive pressure.",
    "Supply chain disruptions resulted in significant production delays and lost revenue opportunities.",
    "Inflationary pressures on input costs are expected to continue to compress gross margins.",
    "Customer churn rates increased significantly, reflecting growing dissatisfaction with products.",
    "The company faces growing competition from well-funded new entrants disrupting its core markets.",
    "The company's products face accelerating obsolescence risk as technology disruption intensifies.",
    "Credit rating agencies downgraded the company's debt to below investment grade.",
    "Material weaknesses in internal controls were identified, raising significant audit concerns.",
    "Revenue visibility deteriorated as backlog declined to its lowest level in five years.",
    "Adverse macro conditions and weakening consumer confidence are expected to reduce demand.",
    "The company's pension liabilities represent a significant and growing long-term financial burden.",
    "Capital requirements are expected to remain elevated, severely constraining free cash flow.",
]

_NEUTRAL_SENTENCES: list[str] = [
    # Filing and compliance boilerplate
    "The company files annual reports on Form 10-K with the Securities and Exchange Commission.",
    "Financial statements are prepared in accordance with generally accepted accounting principles.",
    "The following discussion and analysis should be read in conjunction with the financial statements.",
    "The company operates in three reportable segments: Consumer Products, Enterprise, and Government.",
    "Certain reclassifications have been made to prior period amounts to conform to current presentation.",
    "The company's fiscal year ends on the last Saturday of September each year.",
    "All dollar amounts are in millions unless otherwise indicated.",
    "The consolidated financial statements include the accounts of the company and its subsidiaries.",
    "References to the company mean the registrant and its wholly-owned subsidiaries.",
    "The company was incorporated in the state of Delaware and maintains its principal offices in California.",
    "The company adopted ASU 2023-07 effective the first quarter of the current fiscal year.",
    "Prior period amounts have been restated to reflect the adoption of the new accounting standard.",
    "Segment financial information is presented in accordance with ASC Topic 280.",
    "These financial statements are prepared on a going concern basis.",
    "The company's common stock is listed on the Nasdaq Global Select Market.",
    # Accounting policies
    "Revenue is recognized when control of promised goods or services is transferred to customers.",
    "The company uses the straight-line method to depreciate property, plant and equipment.",
    "Goodwill is not amortized but is tested for impairment at least annually or when indicators arise.",
    "The company measures financial instruments at fair value on a recurring basis.",
    "Stock-based compensation expense is recognized over the requisite service period of the award.",
    "Deferred tax assets and liabilities are measured using enacted tax rates expected to apply.",
    "The company accounts for business combinations using the acquisition method of accounting.",
    "Inventories are stated at the lower of cost or net realizable value on a first-in, first-out basis.",
    "Foreign currency transactions are translated using the exchange rate at the transaction date.",
    "The company capitalizes internal-use software costs incurred during the application development stage.",
    "Operating lease right-of-use assets and liabilities are recognized at the lease commencement date.",
    "The company uses the portfolio approach to assess interest rate risk on its investment portfolio.",
    "Warranty obligations are estimated based on historical experience and expected future costs.",
    "The allowance for doubtful accounts is estimated based on historical collection experience.",
    "The company recognizes revenue on a gross basis when acting as the principal in a transaction.",
    # Operations descriptions
    "The company designs, manufactures and markets smartphones, personal computers and related accessories.",
    "Products are sold through both direct retail channels and indirect third-party distribution.",
    "The company employs approximately 164,000 full-time equivalent employees worldwide.",
    "Research and development activities are conducted at facilities across multiple countries.",
    "The company maintains manufacturing relationships with contract manufacturers primarily in Asia.",
    "Distribution operations are managed through a combination of company-owned and third-party facilities.",
    "The company's products are sold in over 150 countries and territories around the world.",
    "Customer support services are provided through the company's retail stores and online channels.",
    "The company's principal executive offices are located in Cupertino, California.",
    "The board of directors consists of eight independent directors and the Chief Executive Officer.",
    "The company's Audit Committee consists entirely of independent directors.",
    "The company maintains a dual-class share structure as described in the certificate of incorporation.",
    "The company's transfer agent and registrar is Computershare Trust Company.",
    "Capital expenditures for the fiscal year were $11.4 billion.",
    "The company had 5,200 retail store locations in operation at the end of the fiscal year.",
    # Management discussion boilerplate
    "The following discussion contains forward-looking statements that involve risks and uncertainties.",
    "Actual results may differ materially from those described in forward-looking statements.",
    "The company defines non-GAAP earnings per share to exclude certain non-recurring items.",
    "Management uses non-GAAP measures to evaluate operating performance and allocate resources.",
    "The company's critical accounting estimates require significant judgment and complex assumptions.",
    "Results of operations for the periods presented are not necessarily indicative of future results.",
    "The company maintains liquidity sufficient to meet its short-term and long-term obligations.",
    "Diluted weighted average shares outstanding were 15.4 billion for the current fiscal year.",
    "The effective tax rate for the fiscal year was 14.5 percent compared to 13.3 percent in the prior year.",
    "The company has operating lease obligations of $15.4 billion, with a weighted average term of 11 years.",
    "Amortization of acquired intangibles was $312 million for the current period.",
    "The company had $6.3 billion of unrecognized tax benefits at the end of the current fiscal year.",
    "The company's defined benefit pension plans had a funded status of $2.1 billion at year-end.",
    "Outstanding commercial paper had a weighted average interest rate of 5.3 percent at period end.",
    "The company had total debt outstanding of $104 billion at the end of the fiscal year.",
    # Risk factor boilerplate
    "The company's operations are subject to a variety of risks and uncertainties described below.",
    "The following risk factors should be considered when evaluating an investment in the company.",
    "Competition in each of the markets the company operates in is intense and ongoing.",
    "The company is subject to the laws and regulations of numerous jurisdictions worldwide.",
    "Changes in tax laws and regulations could affect the company's effective tax rate.",
    "The company relies on a limited number of suppliers for certain key components and raw materials.",
    "Intellectual property protection is an important element of the company's competitive strategy.",
    "The company's success depends in part on its ability to retain and attract qualified personnel.",
    "Economic conditions in the markets where the company operates can affect demand for products.",
    "The company regularly evaluates potential acquisitions, investments and other strategic transactions.",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_training_corpus() -> tuple[list[str], list[str]]:
    """Assemble and shuffle the labeled training corpus.

    Combines all sentence lists and applies simple augmentation by reversing
    the sentence order on duplicated entries, giving the vectorizer more lexical
    variety without fabricating new labels.

    Returns:
        (texts, labels): parallel lists of sentence strings and their labels.
    """
    raw: list[tuple[str, str]] = (
        [(s, _LABEL_POS) for s in _POSITIVE_SENTENCES]
        + [(s, _LABEL_NEG) for s in _NEGATIVE_SENTENCES]
        + [(s, _LABEL_NEU) for s in _NEUTRAL_SENTENCES]
    )

    # Light augmentation: add lowercased copies to reduce capitalisation bias
    augmented = raw + [(text.lower(), label) for text, label in raw]

    texts, labels = zip(*augmented)
    texts, labels = sk_shuffle(list(texts), list(labels), random_state=42)
    return texts, labels


def _chunk_text(text: str, max_words: int = _CHUNK_WORDS) -> list[str]:
    """Split text into overlapping sentence-boundary-aware chunks.

    Splits on sentence-ending punctuation first, then groups sentences until
    the chunk reaches max_words. Prevents the classifier from seeing only the
    first paragraph of a long filing.

    Args:
        text:      Raw text to split.
        max_words: Target word count per chunk.

    Returns:
        List of non-empty chunk strings.
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current_words: list[str] = []
    current_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_count + len(words) > max_words and current_words:
            chunks.append(" ".join(current_words))
            # 25% overlap: keep the last quarter of words for context continuity
            overlap = max(0, len(current_words) - max_words // 4)
            current_words = current_words[overlap:]
            current_count = len(current_words)
        current_words.extend(words)
        current_count += len(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if c]


def _score_to_label(score: float) -> str:
    """Map a numeric score to a string label using the module threshold."""
    if score > _SCORE_THRESHOLD:
        return _LABEL_POS
    if score < -_SCORE_THRESHOLD:
        return _LABEL_NEG
    return _LABEL_NEU


# ── Main class ────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """Finance-domain sentiment classifier for SEC filing text.

    Train on the built-in corpus by calling ``fit()`` with no arguments, or
    supply your own labeled data to fine-tune on a domain-specific dataset.

    Example::

        analyzer = SentimentAnalyzer()
        analyzer.fit()

        result = analyzer.analyze_text(long_filing_excerpt)
        print(result)   # SentimentResult(score=0.42, label='positive', confidence=0.71)
    """

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        texts: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
    ) -> "SentimentAnalyzer":
        """Train the TF-IDF + Logistic Regression pipeline.

        Args:
            texts:  List of training sentences. If None, the built-in finance
                    corpus is used.
            labels: Parallel list of labels ("positive", "neutral", "negative").
                    Required when texts is provided.

        Returns:
            self, to allow chaining: ``SentimentAnalyzer().fit()``.

        Raises:
            ValueError: If texts is provided without labels or lengths differ.
        """
        if texts is None:
            texts, labels = _build_training_corpus()
        else:
            if labels is None:
                raise ValueError("labels must be provided when texts is supplied.")
            if len(texts) != len(labels):
                raise ValueError(f"texts and labels must be the same length ({len(texts)} vs {len(labels)}).")

        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),   # unigrams + bigrams
                        min_df=1,
                        max_features=8_000,
                        sublinear_tf=True,    # log(1 + tf) dampens term frequency outliers
                        strip_accents="unicode",
                        analyzer="word",
                        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1_000,
                        class_weight="balanced",   # handles any corpus imbalance
                        solver="lbfgs",            # lbfgs uses multinomial by default (sklearn>=1.5)
                        random_state=42,
                    ),
                ),
            ]
        )
        self._pipeline.fit(texts, labels)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, texts: list[str]) -> list[SentimentResult]:
        """Predict sentiment for a batch of short texts.

        Intended for pre-chunked inputs. For full filing text use
        ``analyze_text()``, which handles chunking automatically.

        Args:
            texts: List of text strings to classify.

        Returns:
            List of SentimentResult, one per input text (same order).

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        self._require_fitted()
        proba: np.ndarray = self._pipeline.predict_proba(texts)
        # LR class ordering is alphabetical: negative=0, neutral=1, positive=2
        classes: list[str] = list(self._pipeline.named_steps["clf"].classes_)
        neg_idx = classes.index(_LABEL_NEG)
        pos_idx = classes.index(_LABEL_POS)

        results: list[SentimentResult] = []
        for row in proba:
            score = float(row[pos_idx] - row[neg_idx])
            confidence = float(np.max(row))
            results.append(
                SentimentResult(
                    score=round(score, 4),
                    label=_score_to_label(score),
                    confidence=round(confidence, 4),
                )
            )
        return results

    def analyze_text(self, text: str) -> SentimentResult:
        """Analyze a single (potentially long) SEC filing text.

        Splits the text into overlapping chunks of ~150 words, scores each
        chunk independently, then returns the length-weighted average score.
        This prevents the early paragraphs of a long 10-K from dominating.

        Args:
            text: Raw filing text. May be arbitrarily long.

        Returns:
            A single SentimentResult aggregated across all chunks.

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        self._require_fitted()

        chunks = _chunk_text(text)
        if not chunks:
            return SentimentResult(score=0.0, label=_LABEL_NEU, confidence=0.0)

        chunk_results = self.predict(chunks)

        # Weight by chunk length (longer chunks carry more signal)
        weights = np.array([len(c.split()) for c in chunks], dtype=float)
        weights /= weights.sum()

        avg_score = float(np.dot([r.score for r in chunk_results], weights))
        avg_confidence = float(np.dot([r.confidence for r in chunk_results], weights))

        return SentimentResult(
            score=round(avg_score, 4),
            label=_score_to_label(avg_score),
            confidence=round(avg_confidence, 4),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        """Raise RuntimeError if the model has not been trained yet."""
        if self._pipeline is None:
            raise RuntimeError(
                "SentimentAnalyzer has not been fitted. Call fit() first."
            )


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _SAMPLES = {
        "Strongly positive (earnings beat)": (
            "Apple reported record quarterly revenue of $124.3 billion, up 9 percent year-over-year, "
            "beating analyst expectations of $121.1 billion. Earnings per share of $2.18 exceeded the "
            "consensus estimate of $2.10. Gross margin expanded to 47.5 percent, the highest level in "
            "over a decade. Management raised full-year guidance and announced a new $110 billion share "
            "buyback program, reflecting strong confidence in continued free cash flow generation. "
            "Services revenue grew 17 percent, reaching a new all-time record, and the company's "
            "installed base of active devices reached a new high across all major product categories."
        ),
        "Strongly negative (risk and losses)": (
            "The company reported a net loss of $2.4 billion for the quarter, significantly worse than "
            "the consensus estimate for a loss of $1.1 billion. Revenue declined 18 percent year-over-year "
            "due to severe macroeconomic headwinds and intensifying competitive pressure. Gross margins "
            "contracted 450 basis points as input costs rose sharply. The company faces significant "
            "litigation risk from multiple pending class action lawsuits and ongoing SEC and DOJ regulatory "
            "investigations. Management lowered full-year guidance and announced a restructuring plan "
            "affecting 9,000 employees. The company's ability to continue as a going concern is subject "
            "to substantial uncertainty, and the auditors have issued a qualified opinion."
        ),
        "Neutral (boilerplate disclosure)": (
            "The following information is provided pursuant to Item 7 of Form 10-K. The company's fiscal "
            "year ends on the last Saturday of September. Financial statements are prepared in accordance "
            "with United States generally accepted accounting principles. All amounts are in millions of "
            "dollars unless otherwise stated. Certain reclassifications have been made to prior period "
            "amounts to conform to the current period presentation. The company operates through three "
            "reportable segments as defined under ASC 280. The board of directors consists of nine "
            "members including the Chief Executive Officer. The company's common stock is listed on the "
            "Nasdaq Global Select Market under the symbol AAPL."
        ),
        "Mixed (risk factors with growth language)": (
            "The company delivered strong revenue growth of 14 percent year-over-year, driven by record "
            "performance in the services segment. However, the company faces significant regulatory "
            "uncertainty in key international markets, including ongoing antitrust investigations in "
            "Europe. Supply chain disruptions resulted in $1.2 billion in lost revenue during the quarter. "
            "Despite these headwinds, management expressed confidence in the long-term growth outlook "
            "and raised guidance for the services segment. The company also disclosed material weakness "
            "in internal controls over financial reporting, which management expects to remediate by "
            "the end of the current fiscal year."
        ),
    }

    print("Training SentimentAnalyzer on internal finance corpus...")
    analyzer = SentimentAnalyzer()
    analyzer.fit()
    print("Training complete.\n")
    print("=" * 72)

    for title, text in _SAMPLES.items():
        result = analyzer.analyze_text(text)
        bar_len = int(abs(result.score) * 20)
        direction = "+" if result.score >= 0 else "-"
        bar = direction * bar_len
        print(f"\n{title}")
        print(f"  Score      : {result.score:+.4f}  [{bar:<20}]")
        print(f"  Label      : {result.label}")
        print(f"  Confidence : {result.confidence:.4f}")

    print("\n" + "=" * 72)
    print("Chunk-level breakdown for the mixed sample:")
    mixed_text = _SAMPLES["Mixed (risk factors with growth language)"]
    chunks = _chunk_text(mixed_text)
    chunk_results = analyzer.predict(chunks)
    for i, (chunk, res) in enumerate(zip(chunks, chunk_results), 1):
        preview = chunk[:80].replace("\n", " ")
        print(f"  Chunk {i}: score={res.score:+.4f}  label={res.label:<8}  \"{preview}...\"")
