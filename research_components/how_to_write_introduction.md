# How to write the introduction

The accompanying `introduction.tex` is a paper-style draft that you can adapt. Its paragraphs follow a funnel: application context, technical need, prior work and gap, pipeline proposal, then contributions and results.

## Recommended paragraph sequence

1. **Set the scene:** Explain why in-car speech is difficult (road/HVAC noise, reverberation, playback, multiple occupants). Use in-car datasets to establish that this is a real research domain.
2. **Define the system requirement:** Explain that a useful assistant must identify target speech, detect an activation phrase, interpret a command, and map it to an action. Distinguish ordinary speech activity from target-speaker activity.
3. **Review the extraction literature:** Introduce target speaker extraction, contrast it briefly with speaker-independent separation, and motivate SpEx+ as the baseline. Then identify cabin acoustic mismatch as the specific gap your adaptation addresses.
4. **Connect the downstream stages:** Cite work/datasets for keyword spotting and spoken language understanding. Explain why stage-level success is not equivalent to correct end-to-end activation and action.
5. **State your approach:** Summarize the cascade in one clear sentence, then mention which datasets train or evaluate each stage. Avoid a long implementation inventory here.
6. **Give contributions and results:** List the novel system contribution and report each metric together with its dataset and split. Do not present clean intent accuracy or extraction SI-SDR improvement as full pipeline accuracy.
7. **Close with paper organization:** Briefly map the remaining sections, updating the final sentence to match the actual paper structure.

## Writing checks for this study

- Make the research gap precise: the motivation is vehicle-domain mismatch and missing target-conditioned activation/understanding in a unified pipeline, not a claim that all existing TSE models use the same training corpus or fail in cars.
- Separate **speaker identity**, **speech presence**, **wake-word detection**, **intent classification**, and **action mapping**; they are related but different decisions.
- State exact evaluation conditions beside metrics. Current supplied figures are `+10.98 dB` mean SI-SDR improvement on held-out CAVEMOVE and `99.31%` top-1 on the clean Fluent Speech Commands test set.
- Avoid implying demonstrated passenger rejection or end-to-end action accuracy unless a labeled, end-to-end evaluation supports those claims. Describe those as goals/design intent otherwise.
- Check the camera-ready bibliographic style and required citation commands for your venue. The `.bib` keys are intentionally short and stable (for example, `\cite{spexplus}` and `\cite{cavemove}`).

## Using the files in TeXstudio with Biber

Place `introduction.tex` in your main document with `\input{research_components/introduction}`. Add `\addbibresource{research_components/references.bib}` in the preamble when using `biblatex`, and add `\printbibliography` where the references should appear. Select **Biber** as the bibliography backend in TeXstudio. A minimal setup is:

```latex
\usepackage[backend=biber,style=ieee]{biblatex}
\addbibresource{research_components/references.bib}
```

Compile LaTeX, Biber, then LaTeX twice (or use TeXstudio's build chain configured for Biber). If the conference template requires BibTeX rather than `biblatex`, it will need a BibTeX-specific adaptation.
