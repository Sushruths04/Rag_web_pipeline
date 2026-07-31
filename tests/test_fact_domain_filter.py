from rag_gt.core.types import Fact, Span
from rag_gt.facts.domain_filter import (
    fact_domain_reject_reason,
    fact_has_unresolved_deictic,
    filter_fact_domain,
)


def _fact(text: str, *, doc_id: str = "doc", page: int | None = None) -> Fact:
    spans = []
    if page is not None:
        spans = [
            Span(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_c{page:06d}",
                start_token=0,
                end_token=10,
                page_start=page,
                page_end=page,
            )
        ]
    return Fact(fact_id="F1", text=text, role="rule", supporting_spans=spans)


def test_fact_domain_filter_drops_bibliography_lines():
    fact = _fact(
        "Technical Report AFCRL-72-0164, Air Force Cambridge Research "
        "Laboratories, pp. 10-12."
    )

    assert fact_domain_reject_reason(fact) == "bibliography_or_citation"


def test_fact_domain_filter_drops_captions_and_source_tasks():
    assert fact_domain_reject_reason(_fact("Figure 3.2: Backup diagram.")) == (
        "caption_or_heading"
    )
    assert fact_domain_reject_reason(_fact("Exercises 12")) == (
        "source_task_heading"
    )
    assert fact_domain_reject_reason(
        _fact("Exercise 3.5 Imagine that you are designing a robot to run a maze.")
    ) == "source_task_prompt"
    assert fact_domain_reject_reason(
        _fact("212CHAPTER 8. PLANNING AND LEARNING WITH TABULAR METHODS")
    ) == "caption_or_heading"
    assert fact_domain_reject_reason(
        _fact(
            "Repeat for each step: take action and update the value estimate. "
            "Figure 6.12: Q-learning update procedure."
        )
    ) == "embedded_caption_fragment"
    assert fact_domain_reject_reason(
        _fact(
            "Repeat (for each episode): Initialize S, A. "
            "Take action A, observe R, then choose A using the policy."
        )
    ) == "algorithm_listing_fragment"


def test_fact_domain_filter_drops_named_citation_context():
    assert fact_domain_reject_reason(
        _fact("Barto and Duff (1994) discussed policy evaluation in Monte Carlo algorithms.")
    ) == "named_citation_context"
    assert fact_domain_reject_reason(
        _fact("Barto and Duﬀ(1994) discussed policy evaluation in Monte Carlo algorithms.")
    ) == "named_citation_context"
    assert fact_domain_reject_reason(
        _fact("The results in Figure 8.10 are due to Peng and Williams (1993).")
    ) == "source_figure_attribution"
    assert fact_domain_reject_reason(
        _fact("Reprinted from Peng and Williams (1993).")
    ) == "source_figure_attribution"


def test_fact_domain_filter_drops_unresolved_context_dependencies():
    assert fact_domain_reject_reason(
        _fact(
            "The algorithm discussed in the previous section (Figure 9.1) "
            "converges if its step-size parameter decreases."
        )
    ) == "unresolved_source_reference"
    assert fact_domain_reject_reason(
        _fact(
            "This idea can also be extended to linear function approximation "
            "with binary features."
        )
    ) == "unresolved_source_reference"
    assert fact_domain_reject_reason(
        _fact(
            "If we look closely at that example, the learned function is "
            "neither an action-value nor state-value function."
        )
    ) == "unresolved_source_reference"
    assert fact_domain_reject_reason(
        _fact(
            "A new call could be assigned channel 2, as in the right diagram, "
            "without violating the channel reuse constraint."
        )
    ) == "unresolved_source_reference"
    assert fact_domain_reject_reason(
        _fact(
            "Another way of saying this is that a policy greedy with respect "
            "to the optimal value function is optimal."
        )
    ) == "unresolved_source_reference"
    assert fact_domain_reject_reason(
        _fact(
            "In other words, the method converges to an optimal policy for "
            "playing the game."
        )
    ) == "unresolved_source_reference"
    assert fact_domain_reject_reason(
        _fact(
            "We have just shown that the expected update is equal to the "
            "gradient of expected reward."
        )
    ) == "unresolved_source_reference"
    assert fact_domain_reject_reason(
        _fact(
            "The backup diagram for Samuel's checkers player shows its evaluation structure."
        )
    ) == "visual_only_fact"


def test_fact_domain_filter_keeps_self_contained_claim_with_figure_citation():
    assert fact_domain_reject_reason(
        _fact(
            "Eligibility traces provide faster learning when rewards are delayed "
            "by many steps (see Figure 9.12)."
        )
    ) is None


def test_fact_domain_filter_keeps_content_facts_and_counts_reasons():
    facts = [
        _fact("On-policy control estimates action values while following a policy."),
        _fact("Further Reading"),
        _fact("Pages 44-46"),
    ]

    kept, dropped = filter_fact_domain(facts)

    assert [f.text for f in kept] == [
        "On-policy control estimates action values while following a policy."
    ]
    assert dropped == {
        "reference_heading": 1,
        "page_range_or_locator": 1,
    }


def test_fact_domain_filter_drops_journal_volume_page_ranges():
    assert fact_domain_reject_reason(
        _fact("Artificial Intelligence, 72:81-138.")
    ) == "page_range_or_locator"
    assert fact_domain_reject_reason(
        _fact("Morgan Kaufmann, San Mateo, CA, pages 217-246.")
    ) == "bibliography_or_citation"


def test_fact_domain_filter_drops_configured_noncontent_pages():
    assert fact_domain_reject_reason(
        _fact(
            "Connectionist Problem Solving: Computational Aspects of Biological Learning.",
            doc_id="SuttonBartoIPRLBook2ndEd",
            page=325,
        )
    ) == "excluded_source_section"
    assert fact_domain_reject_reason(
        _fact(
            "A free PDF or low cost print is available at the publisher website.",
            doc_id="Computer-Networking-Principles-Bonaventure_f",
            page=2,
        )
    ) == "excluded_source_section"
    assert fact_domain_reject_reason(
        _fact(
            "Klaus Pohl holds a full professorship at the University of Duisburg-Essen.",
            doc_id="requirementsengi",
            page=2,
        )
    ) == "excluded_source_section"
    assert fact_domain_reject_reason(
        _fact(
            "First, we thank those who helped develop the overall view presented in this book.",
            doc_id="SuttonBartoIPRLBook2ndEd",
            page=10,
        )
    ) == "excluded_source_section"


def test_fact_domain_filter_drops_heading_prefix_fragments():
    assert fact_domain_reject_reason(
        _fact("THE REINFORCEMENT LEARNING PROBLEM with an environment, in this case with an opponent player.")
    ) == "heading_prefix_fragment"
    assert fact_domain_reject_reason(
        _fact("VALUE PREDICTION WITH FUNCTION APPROXIMATION they are trying to approximate.")
    ) == "heading_prefix_fragment"
    assert fact_domain_reject_reason(
        _fact("ELIGIBILITY TRACES Initialize Q(s, a)")
    ) == "heading_prefix_fragment"


def test_fact_domain_filter_drops_dangling_section_references():
    assert fact_domain_reject_reason(
        _fact("Whereas in Monte Carlo backups the target is the return, in 7.1.")
    ) == "dangling_section_reference"


def test_fact_domain_filter_drops_section_prefix_and_imperative_tasks():
    assert fact_domain_reject_reason(
        _fact("9.2 Gradient-descent methods for minimizing mean-squared error are well known.")
    ) == "section_prefix_fragment"
    assert fact_domain_reject_reason(
        _fact("Show the sequence of equations analogous to (3.12), but for action values.")
    ) == "source_task_prompt"


def test_fact_domain_filter_drops_front_matter_artifacts():
    assert fact_domain_reject_reason(
        _fact("A free PDF or low cost print is available at https://example.org/book.")
    ) == "front_matter_artifact"
    assert fact_domain_reject_reason(
        _fact("Translated from German by Thorsten Weyer, Bastian Tenbergen, and Marta Tayeh.")
    ) == "front_matter_artifact"
    assert fact_domain_reject_reason(
        _fact("Klaus is co-author of more than 250 peer-reviewed publications.")
    ) == "author_bio"


def test_fact_domain_filter_drops_release_heading_fragment():
    assert fact_domain_reject_reason(
        _fact(
            "Services and protocols 15 Computer Networking : Principles, Protocols and Practice, Release 0.25",
            doc_id="Computer-Networking-Principles-Bonaventure_f",
            page=20,
        )
    ) == "source_release_header"


def test_fact_domain_filter_keeps_valid_short_complete_fact():
    assert fact_domain_reject_reason(
        _fact("A policy defines agent behavior.")
    ) is None


def test_fact_domain_filter_keeps_comma_ending_fragment_in_pool():
    """Comma-ending facts stay in the pool; they are filtered at chain/quality level."""
    assert fact_domain_reject_reason(
        _fact("of optimal control, such as dynamic programming,")
    ) is None


# ---------- fact_has_unresolved_deictic ----------


def test_deictic_opener_this_can_detected():
    assert fact_has_unresolved_deictic(
        _fact("This can make Monte Carlo methods particularly attractive.")
    )


def test_deictic_opener_it_is_detected():
    assert fact_has_unresolved_deictic(
        _fact("It is because of this that bootstrapping methods diverge.")
    )


def test_deictic_opener_these_allow_detected():
    assert fact_has_unresolved_deictic(
        _fact("These allow the agent to estimate values without a model.")
    )


def test_deictic_opener_they_have_detected():
    assert fact_has_unresolved_deictic(
        _fact("They have lower asymptotic error than bootstrapping methods.")
    )


def test_deictic_opener_suppressed_by_canonical_form():
    """canonical_form populated means the SFU upgrader resolved the reference — allow it."""
    fact = _fact("This can make Monte Carlo methods particularly attractive.")
    fact.canonical_form = "The independence property makes Monte Carlo methods attractive."
    assert not fact_has_unresolved_deictic(fact)


def test_self_contained_opener_not_flagged():
    assert not fact_has_unresolved_deictic(
        _fact("Monte Carlo methods produce independent estimates for each state.")
    )


def test_that_clause_not_flagged():
    """'That' as a conjunction (not a deictic) is not flagged."""
    assert not fact_has_unresolved_deictic(
        _fact("The key insight is that temporal difference methods can bootstrap.")
    )
