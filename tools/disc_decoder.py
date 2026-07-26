"""DISC intervention applied to ReLM target-mask logits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

try:
    from artifact_utils import tokenizer_hash
except ImportError:  # Supports ``python -m tools.eval_relm_disc``.
    from tools.artifact_utils import tokenizer_hash


class DISCDecoder:
    def __init__(
        self,
        artifact_dir: str | Path,
        device: torch.device,
        alpha: float = 1.1,
        beta: float = 0.7,
        copy_punishment: float = 0.0,
    ) -> None:
        artifact_dir = Path(artifact_dir)
        self.metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("protocol") != "paper":
            raise ValueError("Main DISC decoding requires a paper-protocol artifact")
        self.alpha = alpha
        self.beta = beta
        self.copy_punishment = copy_punishment
        self.device = device
        self.char_token_ids = torch.tensor(
            self.metadata["char_token_ids"], dtype=torch.long, device=device
        )
        phonetic = np.asarray(np.load(artifact_dir / "phonetic.npy", mmap_mode="r"), dtype=np.float32)
        glyph = np.asarray(np.load(artifact_dir / "glyph.npy", mmap_mode="r"), dtype=np.float32)
        if phonetic.shape != glyph.shape or phonetic.shape != (
            len(self.char_token_ids),
            len(self.char_token_ids),
        ):
            raise ValueError("DISC artifact matrices do not match char_token_ids")
        if self.char_token_ids.numel() and int(self.char_token_ids.max()) >= self.metadata["num_vocab_tokens"]:
            raise ValueError("DISC artifact contains a token ID outside its recorded vocabulary")
        self.phonetic_matrix = torch.from_numpy(phonetic.copy()).to(device=device)
        self.glyph_matrix = torch.from_numpy(glyph.copy()).to(device=device)
        vocab_size = int(self.metadata["num_vocab_tokens"])
        self.token_id_to_char_row = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        self.token_id_to_char_row[self.char_token_ids] = torch.arange(
            len(self.char_token_ids), dtype=torch.long, device=device
        )
        self.covered_positions = 0
        self.total_positions = 0

    def validate_model_path(self, model_path: str | Path) -> None:
        actual = tokenizer_hash(model_path)
        expected = self.metadata.get("tokenizer_sha256", self.metadata.get("vocab_sha256"))
        if actual != expected:
            raise ValueError(
                "DISC artifact tokenizer hash does not match --model_path; "
                "rebuild the artifact for this exact ReLM tokenizer"
            )

    @property
    def coverage(self) -> float:
        return self.covered_positions / self.total_positions if self.total_positions else 0.0

    def adjust(
        self,
        mask_logits: torch.Tensor,
        source_ids: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return argmax IDs after applying DISC to ``mask_logits``.

        ``source_ids`` is explicitly aligned with the target mask positions;
        it must not contain the target ``[MASK]`` token IDs.
        """

        if mask_logits.ndim != 3 or source_ids.ndim != 2:
            raise ValueError("Expected mask_logits [batch,length,vocab] and source_ids [batch,length]")
        if mask_logits.shape[:2] != source_ids.shape:
            raise ValueError("Source and target position shapes do not match")
        if valid_mask is None:
            valid_mask = torch.ones_like(source_ids, dtype=torch.bool)
        rows = torch.full_like(source_ids, -1)
        in_range = (source_ids >= 0) & (source_ids < len(self.token_id_to_char_row))
        rows[in_range] = self.token_id_to_char_row[source_ids[in_range]]
        valid_disc = valid_mask & (rows >= 0)
        self.covered_positions += int(valid_disc.sum().item())
        self.total_positions += int(valid_mask.sum().item())

        probs = torch.softmax(mask_logits.float(), dim=-1)
        safe_rows = rows.clamp_min(0)
        p_rows = self.phonetic_matrix[safe_rows]
        g_rows = self.glyph_matrix[safe_rows]
        disc_rows = self.beta * p_rows + (1.0 - self.beta) * g_rows
        disc_rows = disc_rows * valid_disc.unsqueeze(-1)
        candidate_probs = probs.index_select(dim=-1, index=self.char_token_ids)
        candidate_scores = candidate_probs + self.alpha * disc_rows
        scores = probs.clone()
        scores[..., self.char_token_ids] = candidate_scores

        if self.copy_punishment > 0:
            safe_source_ids = source_ids.clamp_min(0)
            scores.scatter_add_(
                dim=-1,
                index=safe_source_ids.unsqueeze(-1),
                src=-self.copy_punishment * valid_mask.unsqueeze(-1).to(scores.dtype),
            )
        return scores.argmax(dim=-1)
