from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import List, Optional, Dict, Any, Tuple

from ovos_plugin_manager.stt import STT, load_stt_plugin
from ovos_plugin_manager.utils.audio import AudioData
from ovos_utils import classproperty
from ovos_utils.log import LOG

from ovos_stt_plugin_rover.rover import ROVER



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
        super().__init__(config)

        backend_cfgs = self.config.get("backends", [])
        if not backend_cfgs:
            raise RuntimeError("ROVER STT requires at least one backend")

        self.timeout: float = float(self.config.get("timeout", 10.0))
        self.max_workers: int = int(self.config.get("workers", len(backend_cfgs)))

        self.backends: List[STT] = []
        for bcfg in backend_cfgs:
            module = bcfg.get("module")
            if not module:
                raise RuntimeError("Backend entry missing 'module' key")

            try:
                plugin = load_stt_plugin(module)(bcfg.get("config", {}))
            except Exception as e:
                LOG.error(f"Failed to load backend '{module}': {e}")
                continue
            self.backends.append(plugin)

        if not self.backends:
            raise RuntimeError("All backends failed to load")

        self.rover = ROVER()

    # ----------------------------------------------------------------------

    def _run_backend(
        self, stt: STT, audio: AudioData, language: Optional[str]
    ) -> Tuple[bool, Optional[str], Optional[Exception]]:
        """Isolated execution wrapper for a backend."""
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
        Intersection of all backend supported languages.

        This is a static property; it reads plugin configuration from core config.
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
