from django.db import models


class EvalDataset(models.TextChoices):
    SIGIR = "sigir", "TrialGPT SIGIR 2016"
    TREC_2021 = "trec_2021", "TrialGPT TREC 2021 CT"
    TREC_2022 = "trec_2022", "TrialGPT TREC 2022 CT"
    CHIA = "chia", "Chia"
    N2C2_2018 = "n2c2_2018", "n2c2 2018 Track 1"
    HANDLABEL = "handlabel", "Hand-labelled Synthea pairs"


class EvalSystem(models.TextChoices):
    SUPERVISOR = "supervisor", "LLM supervisor"
    FIXED_PIPELINE = "fixed_pipeline", "Fixed pipeline (ablation)"
    BM25 = "bm25", "BM25 baseline"
    PARSER = "parser", "Criteria Parser alone"
    JUDGE = "judge", "Eligibility Judge alone"


class EvalStatus(models.TextChoices):
    RUNNING = "running", "Running"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class EvalRun(models.Model):
    """One evaluation run: a system on a dataset, with its metrics."""

    dataset = models.CharField(max_length=16, choices=EvalDataset.choices)
    system = models.CharField(max_length=16, choices=EvalSystem.choices)
    status = models.CharField(max_length=16, choices=EvalStatus.choices, default=EvalStatus.RUNNING)
    config = models.JSONField(default=dict, blank=True, help_text="Model, provider, temperature, budgets.")
    snapshot_id = models.CharField(max_length=64, blank=True, help_text="Frozen trial snapshot used.")
    code_version = models.CharField(max_length=64, blank=True, help_text="Git SHA of the backend.")
    metrics = models.JSONField(default=dict, blank=True)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    report_path = models.CharField(max_length=255, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"{self.system} on {self.dataset} ({self.status})"
