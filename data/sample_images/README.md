# Sample Images

This directory holds 5 sample construction-site images used by the demo
notebook and the Gradio UI's "Try Example" buttons.

The images are populated by `data/download_datasets.sh`, which fetches a
selection of Creative-Commons-licensed photos from Wikimedia Commons via
the `Special:FilePath` redirect (the only Commons URL that survives file
renames and rehashes):

| File                              | Source filename on Commons                                                                          | Notes                          |
|-----------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------|
| `construction_workers.jpg`        | *A construction worker plastering a house 01.jpg*                                                    | Worker plastering — PPE check  |
| `scaffolding_workers.jpg`         | *A worker wears a helmet and visor at a Hong Kong construction site during a heatwave.jpg*           | Helmet, hi-vis, visor          |
| `construction_excavation.jpg`     | *Excavation.jpg*                                                                                     | Open excavation                |
| `construction_site_with_ppe.jpg`  | *Construction Worker On Footpath.jpg*                                                                | Single-worker PPE scene        |
| `construction_concrete_pour.jpg`  | *Rebar worker.jpg*                                                                                   | Rebar / formwork               |

All images are CC-licensed and used for illustrative purposes only. The
OSHA citations SiteSafe reports against any sample image are *predictions
of a fine-tuned model*, not findings of fact. They are **not** a
determination that the crew or employer in the source photo committed the
cited violation.

If a download fails (no internet), the rest of SiteSafe still works — the
"Try Example" carousel gracefully degrades when no sample images are
present, and users can always upload their own image, paste from
clipboard, or use a webcam capture.
