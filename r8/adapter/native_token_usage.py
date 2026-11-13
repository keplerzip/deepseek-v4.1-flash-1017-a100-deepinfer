"""Per-request accounting of generated native IDs, independent of text decoding.

Totals describe accepted model generation, including EOS and tokens later hidden
by protocol parsing. Reasoning counts describe the raw generation phase before
its first real thinking-end ID, including leading newlines/control text. They
are not a re-tokenization of visible reasoning. An ambiguous boundary is None.
No tokenizer decode, parser mutation, clipping, or shared request state is used.
"""
from collections.abc import Mapping
import re


def _option(options, name, default=None):
    return options.get(name, default) if isinstance(options, Mapping) else getattr(options, name, default)


def _stage_name(stage):
    if stage is None:
        return None
    name = getattr(stage, 'name', None)
    if isinstance(name, str):
        return name
    text = str(stage)
    for known in ('ReasoningStart', 'ReasoningContinuation', 'ContentContinuation'):
        if re.search(r'(?<![A-Za-z_])' + known + r'(?![A-Za-z_])', text):
            return known
    return 'UNKNOWN'


def _single_id(tokenizer, spelling):
    try:
        ids = list(tokenizer.encode(spelling))
    except Exception:
        return None
    return ids[0] if len(ids) == 1 and type(ids[0]) is int and ids[0] >= 0 else None


class NativeTokenUsage:
    """A fresh instance belongs to one EngineInference generator only.

    configure() is called before any generation; engine_finished is set by the
    owner only after a genuine engine terminal output. Unknown reasoning remains
    unknown even if a later marker is recognized. Total always counts every ID.
    """

    def __init__(self):
        self.configured = False
        self.total = 0
        self.reasoning = None
        self.engine_finished = False
        # Backend facts are recorded before yielding a final token batch. The
        # official parser may emit tool completion events while consuming that
        # batch, before it consumes InferenceChunk.finish().
        self.backend_finish_reason = None
        self.backend_finished = False
        self.engine_request_id = None
        self.prompt_tokens = None
        self.requested_max_tokens = None
        self.effective_max_tokens = None
        self.suffix_audits = set()
        self.phase = 'unconfigured'
        self.unknown_reason = 'not_configured'
        self._end_id = None
        self._dsml_id = None

    @property
    def finished_phase(self):
        return self.phase in ('content', 'eos')

    def configure(self, tokenizer, parsing_options):
        if self.total or self.engine_finished:
            raise RuntimeError('NativeTokenUsage cannot be reused after generation')
        specials = _option(parsing_options, 'special_tokens')
        end = _option(specials, 'thinking_end', '</think>')
        dsml = _option(specials, 'dsml', '｜DSML｜')
        self._end_id = _single_id(tokenizer, end)
        self._dsml_id = _single_id(tokenizer, dsml)
        self.configured = True
        self.unknown_reason = None
        stage = _stage_name(_option(parsing_options, 'thinking_initial_stage'))
        if _option(parsing_options, 'starts_in_tool_calls', False):
            self.phase, self.reasoning = 'content', 0
        elif stage in (None, 'ContentContinuation'):
            self.phase, self.reasoning = 'content', 0
        elif stage in ('ReasoningStart', 'ReasoningContinuation'):
            self.phase, self.reasoning = 'reasoning', 0
            if self._end_id is None or self._dsml_id is None or self._end_id == self._dsml_id:
                self.reasoning = None
                self.unknown_reason = 'special_marker_not_a_distinct_single_id'
        else:
            self.phase, self.reasoning = 'unknown', None
            self.unknown_reason = 'unknown_initial_stage'
        return self

    def ingest_tokens(self, ids, eos_ids=()):
        if not self.configured:
            raise RuntimeError('Configure native usage before ingestion')
        if self.engine_finished:
            raise RuntimeError('Cannot ingest after engine completion')
        tokens = list(ids)
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError('Native token IDs must be non-negative integers')
        eos = {token for token in (() if eos_ids is None else eos_ids)
               if type(token) is int and token >= 0}
        self.total += len(tokens)
        for token in tokens:
            if token in eos:
                self.phase = 'eos'
                continue
            if self.phase != 'reasoning':
                continue
            if token == self._end_id:
                self.phase = 'content'
            elif token == self._dsml_id:
                # The core parser can enter tools before ThinkingEnd. Without
                # its byte-level marker state we must not count tool arguments
                # as reasoning or guess whether DSML was literal content.
                self.reasoning = None
                self.unknown_reason = 'dsml_before_thinking_end'
                self.phase = 'unknown'
            elif self.reasoning is not None:
                self.reasoning += 1
        return self
