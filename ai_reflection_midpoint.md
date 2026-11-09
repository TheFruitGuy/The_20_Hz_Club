# Reflection on GenAI use, Phase I

**AI-aided pair:** Matthias Nagl (k01126326), Klaus Lex (k01027570)

Team: The 20 Hz Club. Human-only pair in Phase I: Abhirup Mitra, Gaurav Parajuli. Details of the
interactions: [`ai-interactions/phase1.md`](ai-interactions/phase1.md).

## How we used it

We worked with Claude (Anthropic) as a chat partner and as a coding agent with access to our repo and
the CP servers. We set the direction: the Phase I questions, which tests to run, and the rules every
analysis had to follow (the same thresholds and matching for all three models, a pre-registered
expectation before every run, a known number to compare every result with). Within that frame we
delegated the implementation: the AI wrote most of the code (setup, prediction export, matching, probes,
figures), ran long jobs on the server, drafted candidate hypotheses for us to choose from and drafted
write-ups that we edited. Nothing went into a table or a slide before we had checked it.

## Breadth and depth

We used the speed mainly to widen the analysis. We compared three models instead of one, trained five
seeds of every NAVE variant we tested, and ran about 40 pre-registered tests in Phase I. For depth we
followed single findings further: from the frequency probe to the drift measurement, the test on
shifted real recordings, the augmentation and its replication over seeds and ablation rungs. The
human-only pair went deeper on fewer questions with far fewer runs, and some of our broadest results
started from their observations (the seeds missing the same calls, the ensemble question), which we then
extended to all classes and all three models.

## Speed and the cost of checking

Follow-up tests took hours instead of days, and part of the saved time went into checking. Two checks
paid off: comparing the fusion results with the known single-model score exposed a float32 issue in
the fusion code that had made NAVE look 0.18 worse, and testing the agreement-based precision estimate
on a labelled dev site showed it was off by 0.3, so we did not use it. Our rule was to accept no number
without something to compare it with.

## Failure patterns and what worked

The AI's typical failures were confident write-ups of exploratory results, numerical bugs that looked
like findings, plausible but wrong mechanisms (close to half of the registered tests were refuted, not
supported or only partly supported) and internal shorthand in text meant for others. What kept the
analysis reliable was our process around it: pre-registration before every run, comparisons with known
numbers, asking for several options and choosing ourselves, and keeping the label question with humans.

## Confidence

We are most confident where a pre-registered test held for every seed (drift, loudness, tile edges) and
least confident about precision, because the dev labels are incomplete. For that question we decided
on a blind listening check by all four of us instead of relying on the models or the AI.

## What we take into Phase II

When we switch to human-only, we expect to cover fewer questions, so we will pick the two with the
strongest Phase I evidence: why all three models expect the old Z-call pitch, and why loud input breaks
NAVE. The habits we keep without AI: pre-register, compare every result with a known number, and look
at the raw audio before trusting a label.
