import os
import logging
from typing import Any, Dict, List, Tuple, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.model_summary import summarize, ModelSummary as Summary
from lightning.pytorch.utilities.model_summary.model_summary import _format_summary_table

log = logging.getLogger(__name__)

class ModelSummary(Callback):
    def __init__(self, max_depth: int = 1, output_dir: str = None, **summarize_kwargs: Any) -> None:
        self._max_depth = max_depth
        self._summarize_kwargs = summarize_kwargs
        self.output_dir = output_dir or './'  # Default to current directory if not provided

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._max_depth:
            return

        model_summary = self._summary(trainer, pl_module)
        summary_data = model_summary._get_summary_data()
        total_parameters = model_summary.total_parameters
        trainable_parameters = model_summary.trainable_parameters
        model_size = model_summary.model_size

        if trainer.is_global_zero:
            summary_text = self.summarize(summary_data, total_parameters, trainable_parameters, model_size, **self._summarize_kwargs)
            self._save_summary_to_file(summary_text)

    def _summary(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Union[Summary, Summary]:
        return summarize(pl_module, max_depth=self._max_depth)

    @staticmethod
    def summarize(
        summary_data: List[Tuple[str, List[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        **summarize_kwargs: Any,
    ) -> str:
        summary_table = _format_summary_table(
            total_parameters,
            trainable_parameters,
            model_size,
            *summary_data,
        )
        return summary_table

    def _save_summary_to_file(self, summary_text: str):
        filename = os.path.join(self.output_dir, 'summary.txt')
        with open(filename, 'w') as f:
            f.write(summary_text)