# Data sources and licences

The Indic-Skill-100 pilot set (`indic-skill-100/`) contains 152 short clips:

| Source | Clips | Terms |
|---|---|---|
| Pexels stock footage | 90 | Pexels licence: free to use; check the Pexels licence page before redistributing the files. |
| YouTube Shorts | 62 | Copyright stays with the original creators. Source links are in `indic-skill-100/yt_sources.csv`. Used for research and evaluation only. Do not redistribute the video files. |

Notes
- The repository does not contain the raw video files, only metadata, scores and extracted features.
- Human technique scores in `scores.csv` were collected for this project. Add how many raters scored each clip and the checklist used before publishing results.
- Before releasing the dataset publicly, confirm the terms for every source and obtain consent from anyone identifiable in the footage.
- The wage model is trained on synthetic data generated in code. It contains no real worker or wage data.
