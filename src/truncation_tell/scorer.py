"""Teacher-forced log-probability scoring.

One primitive underlies everything downstream: log P(response | system, prompt)
under a given model. The probe battery, the baseline, and the attack weights are
all differences of this quantity.
"""

import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    """CUDA if present, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Scorer:
    """Scores responses under one model. Holds weights; construct once, reuse."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=dtype
        )
        self.model.to(self.device)
        self.model.eval()

    def token_length(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _prefix_ids(self, system: str | None, prompt: str) -> torch.Tensor:
        """Chat-format the prefix, falling back to plain text if no template."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # The template may or may not already emit BOS as literal text. Letting
            # the tokenizer add it again silently corrupts every downstream
            # log-probability, so only add specials when the template did not.
            bos = getattr(self.tokenizer, "bos_token", None)
            add_special = not (bos and text.startswith(bos))
            ids = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=add_special
            )["input_ids"]
        else:
            head = f"{system}\n\n" if system else ""
            text = f"{head}{prompt}\n"
            ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    def _response_ids(self, response: str) -> torch.Tensor:
        ids = self.tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        return ids.to(self.device)

    @torch.no_grad()
    def logprob(self, system: str | None, prompt: str, response: str) -> float:
        """Summed log-probability of the response tokens."""
        prefix = self._prefix_ids(system, prompt)
        resp = self._response_ids(response)
        if resp.shape[1] == 0:
            return 0.0
        full = torch.cat([prefix, resp], dim=1)
        logits = self.model(full).logits
        return self._score_from_logits(logits, full, prefix.shape[1])

    @torch.no_grad()
    def logprob_pair(
        self, system: str | None, prompt: str, chosen: str, rejected: str
    ) -> tuple[float, float]:
        """Score both responses against one shared prefix forward pass."""
        prefix = self._prefix_ids(system, prompt)
        out = self.model(prefix, use_cache=True)
        scores = []
        for response in (chosen, rejected):
            resp = self._response_ids(response)
            if resp.shape[1] == 0:
                scores.append(0.0)
                continue
            # Copy the cache: the model mutates it in place, so the second
            # response would otherwise continue from the first one's state.
            cache = copy.deepcopy(out.past_key_values)
            step = self.model(resp, past_key_values=cache, use_cache=True)
            # Only the final prefix logit matters -- it predicts the first
            # response token. Concatenating the whole prefix logits copies tens
            # of megabytes per call at a 100k vocab, all of it discarded by the
            # slice in _score_from_logits.
            logits = torch.cat([out.logits[:, -1:, :], step.logits], dim=1)
            full = torch.cat([prefix[:, -1:], resp], dim=1)
            scores.append(self._score_from_logits(logits, full, 1))
        return scores[0], scores[1]

    @staticmethod
    def _score_from_logits(
        logits: torch.Tensor, full: torch.Tensor, prefix_len: int
    ) -> float:
        """Sum log P over the response positions only.

        Position i's logits predict token i+1, so the logit that predicts the
        first response token sits at index prefix_len - 1.
        """
        logprobs = torch.log_softmax(logits[:, prefix_len - 1 : -1, :].float(), dim=-1)
        targets = full[:, prefix_len:]
        gathered = logprobs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return float(gathered.sum().item())
