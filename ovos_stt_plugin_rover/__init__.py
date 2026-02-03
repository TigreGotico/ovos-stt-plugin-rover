from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import List, Optional, Dict, Any, Tuple

from ovos_plugin_manager.stt import STT, load_stt_plugin
from ovos_plugin_manager.utils.audio import AudioData
from ovos_utils import classproperty
from ovos_utils.log import LOG

from ovos_stt_plugin_rover.rover import ROVER, WeightedROVER, IterativeROVER


class ROVERSTT(STT):
    """
    STT plugin performing consensus transcription via ROVER across multiple
    backend STT engines executed in parallel.

    Config example:

        {
            "timeout": 10,
            "workers": 4,
            "backends": [
                {"module": "ovos-stt-plugin-whisper", "config": {...}},
                {"module": "ovos-stt-plugin-vosk", "config": {...}},
                {"module": "ovos-stt-plugin-deepspeech", "config": {...}}
            ]
        }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ROVERSTT instance by loading configured backend STT plugins, configuring execution parameters, and selecting the consensus algorithm.
        
        Parameters:
            config (Optional[Dict[str, Any]]): Configuration dictionary. Expected keys:
                - "backends" (List[Dict]): Required. Each entry must include:
                    - "module" (str): module path/name of the STT plugin to load.
                    - "config" (Dict, optional): plugin-specific config passed to the plugin constructor.
                    - "weight" (float, optional): weight used by WeightedROVER (default 1.0).
                - "timeout" (float, optional): per-backend timeout in seconds (default 10.0).
                - "workers" (int, optional): maximum parallel worker threads (default number of backends).
                - "algo" (str, optional): consensus algorithm to use; supported values:
                    - "wROVER" → WeightedROVER
                    - "itROVER" → IterativeROVER
                    - any other value → ROVER (default)
        
        Raises:
            RuntimeError: If "backends" is missing or empty, if a backend entry lacks a "module" key,
                          or if all configured backends fail to load.
        """
        super().__init__(config)

        backend_cfgs = self.config.get("backends", [])
        if not backend_cfgs:
            raise RuntimeError("ROVER STT requires at least one backend")

        self.timeout: float = float(self.config.get("timeout", 10.0))
        self.max_workers: int = int(self.config.get("workers", len(backend_cfgs)))

        self.backends: List[STT] = []
        weights = []

        for bcfg in backend_cfgs:
            module = bcfg.get("module")
            if not module:
                raise RuntimeError("Backend entry missing 'module' key")

            try:
                plugin = load_stt_plugin(module)(bcfg.get("config", {}))
                self.backends.append(plugin)
                # Collect weights from config for WeightedROVER
                weights.append(bcfg.get("weight", 1.0))
            except Exception as e:
                LOG.error(f"Failed to load backend '{module}': {e}")
                continue

        if not self.backends:
            raise RuntimeError("All backends failed to load")

        # Select Algorithm
        algo = self.config.get("algo", "ROVER").lower()
        if algo == "wROVER":
            self.rover = WeightedROVER(weights=weights)
        elif algo == "itROVER":
            self.rover = IterativeROVER()
        else:
            self.rover = ROVER()

    # ----------------------------------------------------------------------

    def _run_backend(
            self, stt: STT, audio: AudioData, language: Optional[str]
    ) -> Tuple[bool, Optional[str], Optional[Exception]]:
        """
            Run a single backend STT and return its outcome.
            
            Parameters:
                stt (STT): Backend STT instance to invoke.
                audio (AudioData): Audio to transcribe.
                language (Optional[str]): Optional language hint for the backend.
            
            Returns:
                Tuple[bool, Optional[str], Optional[Exception]]: A tuple of (ok, transcript, error) where
                    `ok` is True if the backend produced a non-empty transcription,
                    `transcript` is the stripped transcription string or None,
                    and `error` is the exception caught from the backend or None.
            """
        try:
            text = stt.execute(audio, language)

            if isinstance(text, str) and text.strip():
                return True, text.strip(), None

            return False, None, None

        except Exception as exc:
            return False, None, exc

    # ----------------------------------------------------------------------

    def execute(self, audio: AudioData, language: Optional[str] = None) -> str:
        """
        Run all STT backends in parallel with timeouts and error isolation.
        Aggregate the successful transcripts via ROVER.
        """

        futures = []
        results: List[str] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for stt in self.backends:
                fut = executor.submit(self._run_backend, stt, audio, language)
                futures.append(fut)

            for fut in as_completed(futures, timeout=self.timeout):
                try:
                    ok, text, err = fut.result(timeout=self.timeout)
                except TimeoutError:
                    LOG.warning("STT backend timed out")
                    continue
                except Exception as exc:
                    LOG.warning(f"STT backend failed: {exc}")
                    continue

                if err is not None:
                    LOG.warning(f"STT backend raised exception: {err}")
                    continue

                if ok and text:
                    results.append(text)

        if not results:
            LOG.warning("No backend produced a valid transcript")
            return ""

        if len(results) == 1:
            return results[0]

        return self.rover.fit(results)

    # ----------------------------------------------------------------------

    @classproperty
    def available_languages(cls) -> set:
        """
        Compute the set of languages supported by all configured STT backends.
        
        Reads the "stt.backends" entries from core configuration and returns the intersection of each backend's `available_languages`. If no backends are configured or there is no common language, returns an empty set.
        
        Returns:
            set: Language identifiers supported by every configured backend (empty if none).
        """
        cfg = cls.config_core.get("stt", {}).get("backends", [])
        if not cfg:
            return set()

        langs: Optional[set] = None

        for bcfg in cfg:
            module = bcfg.get("module")
            if not module:
                continue

            try:
                plugin_cls = load_stt_plugin(module)
                plugin_langs = plugin_cls.available_languages
            except Exception:
                continue

            if langs is None:
                langs = set(plugin_langs)
            else:
                langs &= set(plugin_langs)

        return langs or set()


if __name__ == "__main__":
    b = ROVERSTT(config={"lang": "en",
                         "algo": "wROVER",
                         "backends": [
                             {"module": "ovos-stt-plugin-onnxasr",
                              "weight": 0.9,
                              "config": {"model": "nemo-canary-1b-v2", "quantization": "int8"}},
                             {"module": "ovos-stt-plugin-onnxasr",
                              "weight": 0.8,
                              "config": {"model": "nemo-parakeet-tdt-0.6b-v3", "quantization": "int8"}},
                             {"module": "ovos-stt-plugin-vosk",
                              "weight": 0.5,
                              "config": {"lang": "en"}},
                             {"module": "ovos-stt-plugin-citrinet",
                              "weight": 0.3,
                              "config": {"lang": "en"}}
                         ]
                         })

    eu = "/home/miro/PycharmProjects/ovos-stt-plugin-vosk/jfk.wav"
    audio = AudioData.from_file(eu)

    a = b.execute(audio, language="en")
    print(a)