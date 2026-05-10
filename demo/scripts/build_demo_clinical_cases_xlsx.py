"""One-off helper to regenerate demo_clinical_cases.xlsx — run from repo root."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROWS = [
    {
        "case_id": "DEMO-01",
        "true_label": "ak",
        "true_label_display": "Actinic keratosis",
        "clinical_description": (
            "62-year-old male patient, Fitzpatrick type III, with an 8×5 mm rough scaly "
            "lesion on the face near the cheek. The lesion hurts occasionally and has been "
            "bleeding after shaving. Long-term outdoor work exposure. No skin cancer history. "
            "Patient denies alcohol consumption."
        ),
    },
    {
        "case_id": "DEMO-02",
        "true_label": "bcc",
        "true_label_display": "Basal cell carcinoma",
        "clinical_description": (
            "58-year-old female patient, Fitzpatrick type II, with a 7×5 mm pearly papule "
            "on the nose. The lesion has been growing slowly over 6 months and shows "
            "small blood vessels. It is slightly elevated. No history of cancer. "
            "Non-smoker; no pesticide exposure reported."
        ),
    },
    {
        "case_id": "DEMO-03",
        "true_label": "benign_other",
        "true_label_display": "Benign (other)",
        "clinical_description": (
            "34-year-old female patient, Fitzpatrick type IV, with a 4×3 mm flat tan macule "
            "on the upper extremity (forearm). The lesion has not changed in years and does "
            "not itch or hurt. No elevation. No cancer history, no skin cancer history. "
            "Social alcohol consumption on weekends."
        ),
    },
    {
        "case_id": "DEMO-04",
        "true_label": "melanoma",
        "true_label_display": "Melanoma",
        "clinical_description": (
            "51-year-old male patient, Fitzpatrick type I, with an 11×8 mm dark irregular "
            "lesion on the lower extremity (leg). The lesion has changed in colour and "
            "grew rapidly over 2 months. It itches. Previous melanoma history noted. "
            "Patient has a history of smoking."
        ),
    },
    {
        "case_id": "DEMO-05",
        "true_label": "nevus",
        "true_label_display": "Nevus",
        "clinical_description": (
            "28-year-old female patient, Fitzpatrick type III, with a 5×4 mm uniformly "
            "brown symmetric lesion on the trunk. Stable appearance for 10 years; no itch, "
            "no bleed, no change. Not elevated. No cancer history."
        ),
    },
    {
        "case_id": "DEMO-06",
        "true_label": "scc",
        "true_label_display": "Squamous cell carcinoma",
        "clinical_description": (
            "72-year-old male patient, Fitzpatrick type II, with a 12×9 mm ulcerated "
            "plaque on the scalp. The lesion hurts, bleeds easily, and has enlarged over "
            "4 months. History of cancer (internal malignancy) in the past. Former smoker."
        ),
    },
    {
        "case_id": "DEMO-07",
        "true_label": "seborrheic_keratosis",
        "true_label_display": "Seborrheic keratosis",
        "clinical_description": (
            "65-year-old female patient, Fitzpatrick type V, with a 9×6 mm stuck-on "
            "warty lesion on the trunk. Mild itch only; no major change this year. "
            "Slightly elevated. No skin cancer history. Uses piped water at home."
        ),
    },
    {
        "case_id": "DEMO-08",
        "true_label": "bcc",
        "true_label_display": "Basal cell carcinoma",
        "clinical_description": (
            "45-year-old male patient, Fitzpatrick type II, with a 6×4 mm lesion on the trunk. "
            "The lesion has been itching and has changed in appearance over the past 3 months. "
            "It is slightly elevated. Patient has a history of smoking. No cancer history."
        ),
    },
    {
        "case_id": "DEMO-09",
        "true_label": "melanoma",
        "true_label_display": "Melanoma",
        "clinical_description": (
            "49-year-old female patient, Fitzpatrick type II, with a 14×10 mm asymmetric "
            "dark brown-black lesion on the lower extremity. Marked change in borders over "
            "8 weeks; pruritus present. Elevated center. No alcohol consumption. "
            "Negative for pesticide exposure."
        ),
    },
    {
        "case_id": "DEMO-10",
        "true_label": "ak",
        "true_label_display": "Actinic keratosis",
        "clinical_description": (
            "70-year-old male patient, Fitzpatrick type III, multiple rough 5 mm patches on "
            "the neck; focal 6×5 mm lesion on the neck is symptomatic with itch. Sun damage "
            "history. Rural sanitation without mains sewage. Denies bleeding."
        ),
    },
]

# Column order for presenters
COLUMNS = [
    "case_id",
    "true_label",
    "true_label_display",
    "clinical_description",
]


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "demo" / "demo_clinical_cases.xlsx"
    df = pd.DataFrame(ROWS)[COLUMNS]
    readme = pd.DataFrame(
        {
            "instruction": [
                "Copy clinical_description into the Streamlit demo (or notebook) free-text box.",
                "true_label is the intended disease class for your slide only — the app does not read this column.",
                "Classes match Dataset/label_mapping_multiclass.json (ak, bcc, benign_other, melanoma, nevus, scc, seborrheic_keratosis).",
                "Pair each row with a dermoscopy image that matches the scenario when demonstrating predictions.",
                "Regenerate this file: python3 demo/scripts/build_demo_clinical_cases_xlsx.py",
            ]
        }
    )
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="clinical_cases")
        readme.to_excel(writer, index=False, sheet_name="how_to_use")
    print(f"Wrote {out} ({len(df)} rows + how_to_use sheet)")


if __name__ == "__main__":
    main()
