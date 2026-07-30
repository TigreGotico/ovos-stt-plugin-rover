# OVOS ROVER Aggregation STT Plugin

This plugin is a meta-STT engine for OVOS. It runs several STT backends in parallel and combines their output with the ROVER (Recognizer Output Voting Error Reduction) algorithm. The result is a single, more accurate transcript, at the cost of running more than one STT engine per utterance.

## What is ROVER?

ROVER is a post-processing method from [J. G. Fiscus (ASRU 1997)](https://people.csail.mit.edu/joe/sctk-1.2/doc/rover/rover.htm). It aligns multiple ASR hypotheses with a dynamic-programming sequence alignment algorithm, builds a Word Transition Network (WTN), and takes a majority vote at each aligned position. The vote produces one consensus transcript.

ROVER reduces the effect of a single backend's errors:

* If one backend misrecognizes a word but the others agree, the majority vote corrects it.
* If the hypotheses disagree on word order, the alignment step compensates for insertions, deletions, and substitutions.

ROVER helps most when the backends have different error patterns, for example one acoustic model that handles noisy speech well and another that handles accents well.

## Why use this plugin?

This plugin runs any number of OVOS STT plugins in parallel and merges their results into one transcript.

Benefits:

* Higher transcription accuracy than a single backend.
* Less sensitivity to any one backend's errors.
* Works with any mix of engines (Whisper, Vosk, or other OVOS STT plugins).
* Backends run in parallel, so total time is close to the slowest backend, not the sum of all backends.
* Each backend has its own timeout, so one slow or failing backend does not block the others.

Costs:

* More compute, since N STT engines run instead of one.
* More memory use.
* Latency bound by the slowest backend, unless you set an aggressive timeout.

Use this plugin where accuracy matters more than raw latency or compute cost, for example assistants in noisy environments, smart home devices, call-center transcription, or offline transcription services.

## Installation

```bash
pip install ovos-stt-plugin-rover
```

Install the backend STT plugins you want to aggregate as well.

## Configuration example

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
| ---------- | ----------------------------------------------- |
| `timeout`  | Total timeout for all backends to finish        |
| `workers`  | Maximum number of parallel threads              |
| `backends` | List of STT backends to load and aggregate with ROVER |

You can load any STT backend that OVOS supports here.

## How it works

1. The plugin loads all configured STT backends.
2. A thread pool runs each backend in parallel on the same audio.
3. Each backend has its own timeout and exception handling.
4. The plugin collects the transcripts that succeed.
5. If only one transcript succeeds, the plugin returns it directly. If more than one succeeds, they go to the ROVER engine.
6. ROVER aligns the sequences, builds a WTN, and takes a majority vote.
7. The plugin returns the consensus transcript to OVOS.

## When to use this plugin

Use this plugin when:

* Accuracy matters more than compute cost.
* You have multiple ASR models with different strengths.
* You need robustness against a single backend's failures.
* You want ensemble behavior without changing the backend models.

Avoid it when:

* You run on resource-constrained hardware.
* You need low-latency responses.

## Related projects

This plugin loads other [OVOS STT plugins](https://github.com/orgs/OpenVoiceOS/repositories?q=ovos-stt-plugin) as backends, including [ovos-stt-plugin-whisper](https://github.com/OpenVoiceOS/ovos-stt-plugin-whisper) and [ovos-stt-plugin-vosk](https://github.com/OpenVoiceOS/ovos-stt-plugin-vosk), shown in the configuration example above.

## License

Apache-2.0. See [LICENSE](LICENSE).
