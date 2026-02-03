# OVOS ROVER Aggregation STT Plugin

This plugin provides a **meta-STT engine** for OVOS that aggregates the output of multiple individual STT backends using the **ROVER** (Recognizer Output Voting Error Reduction) algorithm.
It delivers **higher transcription accuracy** at the cost of **additional compute**, making it useful in environments where correctness is prioritized over latency or energy use.

---

## What is ROVER?

**ROVER** is a post-processing method introduced by [J. G. Fiscus (ASRU 1997)](https://people.csail.mit.edu/joe/sctk-1.2/doc/rover/rover.htm).
It aligns multiple ASR hypotheses using a dynamic-programming sequence alignment algorithm, constructs a **Word Transition Network (WTN)**, and performs **majority voting** at each aligned position to produce a consensus output.

ROVER reduces the impact of individual backend biases or failure modes:

* If one backend misrecognizes a word but the others agree, the majority vote corrects it.
* If multiple hypotheses disagree, the alignment layer compensates for insertions, deletions, and substitutions.

In practice, ROVER improves accuracy most when backends have **different error characteristics** (e.g., one acoustic model excels at noisy speech, another at accents, another at punctuation).

---

## Why Use This Plugin?

This plugin allows you to run **any number of OVOS STT plugins in parallel** and merge their results into a single transcript.

### ✔ Benefits

* Higher transcription accuracy
* Robust to backend-specific errors
* Supports arbitrary mix of engines (Whisper, Vosk, DeepSpeech, external APIs, etc.)
* Parallel execution for maximal throughput
* Timeout and failure isolation per backend

### ✘ Costs

* More compute (N STT engines run instead of 1)
* Possibly higher memory use
* Increased latency proportional to the slowest backend (unless using aggressive timeouts)

If accuracy is critical (assistants in noisy environments, smart home devices, call-center AI, transcription services) the gain can be significant.

---

## Installation

```bash
pip install ovos-stt-plugin-rover
```

Ensure that the backend STT plugins you want to aggregate are installed as well.

---

## Configuration Example

Add this to your `mycroft.conf`:

```json
{
  "stt": {
    "module": "ovos-stt-plugin-rover",
    "ovos-stt-plugin-rover": {
        "timeout": 12,
        "workers": 4,
        "backends": [
          {
            "module": "ovos-stt-plugin-whisper",
            "config": {
              "model": "small.en"
            }
          },
          {
            "module": "ovos-stt-plugin-vosk",
            "config": {
              "model": "vosk-model-en-us"
            }
          },
          {
            "module": "ovos-stt-plugin-xxx",
            "config": {
              "model": "..."
            }
          }
        ]
    }
  }
}
```

### Fields

| Key        | Meaning                                        |
| ---------- | ---------------------------------------------- |
| `timeout`  | Total timeout for all backends to complete     |
| `workers`  | Max number of parallel threads                 |
| `backends` | List of STT backends to load & ROVER-aggregate |

Any STT backend supported by OVOS can be loaded here.

---

## How It Works

1. The plugin loads all configured STT backends.
2. A thread pool executes each backend in parallel on the same audio.
3. Each backend has isolated timeout and exception handling.
4. Successful transcripts are collected.
5. If only one transcript succeeds → returned directly.
   If multiple → passed to the internal ROVER engine.
6. ROVER aligns the sequences, constructs a WTN, and performs majority voting.
7. The consensus transcript is returned to OVOS.

---

## When Should You Use This?

Use ROVER-STT when:

* **Accuracy is more important than compute cost**
* The system has **multiple ASR models with complementary strengths**
* You need **robustness** against individual backend failures
* You want **ensemble behavior** without modifying backend models

Avoid it when:

* You are running on **resource-constrained hardware**
* You require **low-latency** responses

