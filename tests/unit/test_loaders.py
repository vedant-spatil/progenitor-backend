import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from data.loaders import eval_data
from data.loaders.common import SOURCE_TAG_SYSTEM
from data.loaders.fhir import add_source_tag, put_entry
from data.loaders.synthea import UnresolvedReferenceError, identifier_index, to_put_entries
from data.synthea.generate import keep_module, load_cohort, synthea_command

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

ORG_IDENT = "https://github.com/synthetichealth/synthea|org-ident-1"


def _info_bundle() -> dict[str, Any]:
    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "fullUrl": "urn:uuid:org-1",
                "resource": {
                    "resourceType": "Organization",
                    "id": "org-1",
                    "identifier": [{"system": ORG_IDENT.split("|")[0], "value": "org-ident-1"}],
                },
                "request": {
                    "method": "POST",
                    "url": "Organization",
                    "ifNoneExist": f"identifier={ORG_IDENT}",
                },
            }
        ],
    }


def _patient_bundle() -> dict[str, Any]:
    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "fullUrl": "urn:uuid:pat-1",
                "resource": {"resourceType": "Patient", "id": "pat-1"},
                "request": {"method": "POST", "url": "Patient"},
            },
            {
                "fullUrl": "urn:uuid:enc-1",
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "subject": {"reference": "urn:uuid:pat-1"},
                    "serviceProvider": {"reference": f"Organization?identifier={ORG_IDENT}"},
                },
                "request": {"method": "POST", "url": "Encounter"},
            },
        ],
    }


class TestSyntheaRewrite:
    def test_entries_become_put_with_literal_refs(self) -> None:
        index = identifier_index([_info_bundle()])
        entries = to_put_entries(_patient_bundle(), index)
        assert [e["request"] for e in entries] == [
            {"method": "PUT", "url": "Patient/pat-1"},
            {"method": "PUT", "url": "Encounter/enc-1"},
        ]
        enc = entries[1]["resource"]
        assert enc["subject"]["reference"] == "Patient/pat-1"
        assert enc["serviceProvider"]["reference"] == "Organization/org-1"

    def test_every_resource_tagged_with_source(self) -> None:
        entries = to_put_entries(_patient_bundle(), identifier_index([_info_bundle()]))
        for e in entries:
            assert {"system": SOURCE_TAG_SYSTEM, "code": "synthea"} in e["resource"]["meta"]["tag"]

    def test_no_urn_or_conditional_refs_remain(self) -> None:
        entries = to_put_entries(_patient_bundle(), identifier_index([_info_bundle()]))
        text = json.dumps(entries)
        assert "urn:uuid:" not in text  # references and fullUrls are both rewritten
        assert "?identifier=" not in text

    def test_unknown_conditional_ref_fails_loudly(self) -> None:
        with pytest.raises(UnresolvedReferenceError):
            to_put_entries(_patient_bundle(), identifier_index([]))


def test_source_tag_added_once() -> None:
    res: dict[str, Any] = {"resourceType": "Patient", "id": "x"}
    add_source_tag(res, "mimic")
    add_source_tag(res, "mimic")
    assert res["meta"]["tag"] == [{"system": SOURCE_TAG_SYSTEM, "code": "mimic"}]
    assert put_entry(res)["request"] == {"method": "PUT", "url": "Patient/x"}


class TestSyntheaConfig:
    def test_cohort_totals_200_across_six_conditions(self) -> None:
        cohort = load_cohort()
        names = [c["name"] for c in cohort["conditions"]]
        assert names == ["diabetes", "ckd", "breast_cancer", "lung_cancer", "hypertension", "asthma"]
        assert sum(c["patients"] for c in cohort["conditions"]) == 200
        assert len({c["seed"] for c in cohort["conditions"]}) == 6

    def test_keep_module_matches_any_code(self) -> None:
        codes = [{"system": "SNOMED-CT", "code": "44054006", "display": "Diabetes"}]
        module = keep_module(codes)
        cond = module["states"]["Initial"]["conditional_transition"][0]
        assert cond == {
            "transition": "Keep",
            "condition": {"condition_type": "Active Condition", "codes": codes},
        }

    def test_command_pins_seed_date_and_transaction_bundles(self, tmp_path: Path) -> None:
        cohort = load_cohort()
        cond = cohort["conditions"][2]  # breast_cancer, gender F
        cmd = synthea_command(tmp_path / "s.jar", cohort, cond, tmp_path, tmp_path / "k.json")
        assert cmd[cmd.index("-s") + 1] == str(cond["seed"])
        assert cmd[cmd.index("-r") + 1] == cohort["reference_date"]
        assert cmd[cmd.index("-g") + 1] == "F"
        assert "--exporter.fhir.transaction_bundle=true" in cmd
        assert "--generate.only_alive_patients=true" in cmd


@pytest.mark.django_db
def test_ctgov_study_maps_to_trial() -> None:
    from data.loaders.ctgov import upsert_trial

    study = json.loads((FIXTURES / "ctgov_NCT04280705.json").read_text())
    trial, created = upsert_trial(study)
    assert created
    assert trial.nct_id == "NCT04280705"
    assert trial.overall_status == "COMPLETED"
    assert trial.phases == ["PHASE3"]
    assert trial.conditions == ["COVID-19"]
    assert trial.eligibility_text.startswith("Inclusion Criteria:")
    assert (trial.minimum_age, trial.maximum_age, trial.sex) == ("18 Years", "99 Years", "ALL")
    assert str(trial.source_last_updated) == "2022-03-14"
    _, created_again = upsert_trial(study)
    assert not created_again


def test_eval_cohort_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_data, "EVAL_DIR", tmp_path)
    root = tmp_path / "trialgpt" / "toy"
    (root / "qrels").mkdir(parents=True)
    (root / "queries.jsonl").write_text('{"_id": "q1", "text": "a"}\n{"_id": "q2", "text": "b"}')
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\tNCT1\t2\nq1\tNCT2\t0\n")
    stats = eval_data.cohort_stats("toy", expected=2)
    assert (stats.patients, stats.qrels_patients, stats.qrels_rows, stats.trials) == (2, 1, 2, 2)
    assert stats.labels == {"0": 1, "2": 1}


def test_chia_stats_counts_trials_entities_relations(tmp_path: Path) -> None:
    path = tmp_path / "chia.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for nct in ("NCT00000001", "NCT00000002"):
            for part in ("inc", "exc"):
                zf.writestr(f"{nct}_{part}.txt", "text")
                zf.writestr(
                    f"{nct}_{part}.ann",
                    "T1\tCondition 0 4\tfoo\nT2\tDrug 5 8\tbar\nR1\tHas Arg1:T1 Arg2:T2\n",
                )
        zf.writestr("annotation.conf", "")
    stats = eval_data.chia_stats(path, expected=2)
    assert (stats.trials, stats.entities, stats.relations) == (2, 8, 4)


def test_eval_manifest_pins_expected_counts() -> None:
    manifest = eval_data.load_manifest()
    cohorts = manifest["trialgpt"]["cohorts"]
    assert {k: v["expected_patients"] for k, v in cohorts.items()} == {
        "sigir": 59,
        "trec_2021": 75,
        "trec_2022": 50,
    }
    assert manifest["chia"]["expected_trials"] == 1000
    assert len(manifest["trialgpt"]["commit"]) == 40
