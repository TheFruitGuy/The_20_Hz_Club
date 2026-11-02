# Reflection on GenAI use, Phase I

**AI-aided pair:** Matthias Nagl (k01126326), Klaus Lex (k01027570)

Team: The 20 Hz Club. Human-only pair in Phase I: Abhirup Mitra, Gaurav Parajuli. Details of the
interactions: [`ai-interactions/phase1.md`](ai-interactions/phase1.md).

## What we used it for

We used Claude (Anthropic) as a chat partner and as a coding agent with access to our repo and the CP
servers. It wrote most of the code (setup, prediction export, matching, probes, figures), proposed most
of the hypotheses and tests, ran long jobs on the server and drafted the write-ups. We decided what to
test, wrote every expectation into `logs/preregistration.md` before the run, and checked the results.

## Breadth and depth

The AI mostly changed the breadth. We compared three models instead of one, trained five seeds of every NAVE
variant we tested, and ran about 40 pre-registered tests in Phase I. Depth came from following
single findings further: the frequency probe led to the drift measurement, the shifted-audio test, the
augmentation and its replication over seeds and ablation rungs. The human-only pair went deeper on fewer
questions with far fewer runs, and some of our broadest results started from their observations (the
seeds missing the same calls, the ensemble question).

## Speed and the cost of checking

Follow-up tests took hours instead of days. The saved time partly went into checking: a float32 issue
in the fusion code made NAVE look 0.18 worse until we compared it with the known single-model score,
and the agreement-based precision estimate looked convincing but was off by 0.3. The rule that helped
most was to never accept a number without something to compare it with.

## Confidence

We are most confident where a pre-registered test held for every seed (drift, loudness, tile edges) and
least confident about precision, because the dev labels are incomplete. For that question we chose a
blind listening check by all four of us instead of trusting the models or the AI.

## What we take into Phase II

When we switch to human-only, we expect to cover fewer questions, so we will pick the two with the
strongest Phase I evidence: why all three models expect the old Z-call pitch, and why loud input breaks
NAVE. The habits we keep without AI: pre-register, compare every result with a known number, and look
at the raw audio before trusting a label.
