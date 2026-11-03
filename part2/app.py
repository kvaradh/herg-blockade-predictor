"""
hERG blockade predictor - Gradio web app.

Loads the exact ONNX file submitted for Part 1 and reproduces its predictions:
featurize() and THRESHOLD below are copied verbatim from Part1_[GROUP].py, so the
app and the submitted model agree by construction.
"""
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors, GraphDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
BASIC_N = Chem.MolFromSmarts(
    "[NX3;!$(N[#6]=[O,N,S]);!$(N[S,P]=O);!$(N=*);!$(N#*);!$(N[a]);!$([N+]);!$(N[N,O])]")
CATION_N = Chem.MolFromSmarts("[$([NX4+]);!$([NX4+][a])]")
ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
HALO = Chem.MolFromSmarts("[F,Cl,Br,I]")
N_DESC, N_STR = 25, 14
N_FEATURES = N_DESC + N_STR
_LOWER = set("cnosp")


def _largest(mol):
    frags = Chem.GetMolFrags(mol, asMols=False)
    if len(frags) <= 1:
        return mol, 1
    pieces = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(pieces, key=lambda m: m.GetNumHeavyAtoms()), len(pieces)


def _row(mol, nfrag, charge):
    return (Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol), rdMolDescriptors.CalcExactMolWt(mol),
            Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), Lipinski.NumAromaticRings(mol),
            Lipinski.RingCount(mol), Lipinski.FractionCSP3(mol), Lipinski.NumRotatableBonds(mol),
            mol.GetNumHeavyAtoms(), Lipinski.NumAliphaticRings(mol), Crippen.MolMR(mol),
            len(mol.GetSubstructMatches(BASIC_N)) + len(mol.GetSubstructMatches(CATION_N)),
            Lipinski.NumAromaticHeterocycles(mol), Lipinski.NumSaturatedRings(mol),
            Lipinski.NumHeteroatoms(mol), rdMolDescriptors.CalcLabuteASA(mol),
            GraphDescriptors.HallKierAlpha(mol), GraphDescriptors.Kappa2(mol),
            GraphDescriptors.Chi1v(mol),
            len(mol.GetSubstructMatches(HALO)), len(mol.GetSubstructMatches(ACID)),
            float(nfrag), float(charge), float(charge > 0))


def _string_row(s):
    n = len(s)
    if n == 0:
        return (0.0,) * N_STR
    lower = sum(1 for ch in s if ch in _LOWER)
    return (float(n), float(s.count("N") + s.count("n")), float(s.count("O") + s.count("o")),
            float(s.count("S") + s.count("s")), float(s.count("F")), float(s.count("l")),
            float(s.count("r")), float(s.count("I")), lower / n,
            float(sum(ch.isdigit() for ch in s)), float(s.count("(")),
            float(s.count("+")), float(s.count("-")), float(s.count(".")))


def featurize(smiles):
    rows = np.zeros((len(smiles), N_FEATURES), np.float64)
    for i, smile in enumerate(smiles):
        try:
            text = smile if isinstance(smile, str) else ""
            rows[i, N_DESC:] = _string_row(text)
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                continue
            charge = Chem.GetFormalCharge(mol)
            core, nfrag = _largest(mol)
            rows[i, :N_DESC] = _row(core, nfrag, charge)
        except Exception:
            continue
    return np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)

from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

# TODO 1: your group's names -> Part1_FIRST1-LAST1-FIRST2-LAST2.onnx
NAME = "Armine-Lalani-Awani-Gadre-Faris-Raza-Kavin-Varadharajulu"

# TODO 2: your own threshold; justify it in README.md
THRESHOLD = 0.53

THRESHOLD = 0.53
# --------------------------------------------------------------------------------

ONNX_PATH = next(Path(__file__).parent.glob("Part1_*.onnx"))
SESSION = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name

MAX_LENGTH = 400          # generous; the held-out set goes up to 200 characters
BLOCK, SAFE = "#c5283d", "#1f7a4d"


def validate(text):
    """Called on every keystroke. Returns (message, molecule_or_None).

    Never raises: an exception here would leave the button stuck in its previous
    state, which is worse than reporting the input as unusable.
    """
    try:
        text = (text or "").strip()
        if not text:
            return "Enter a SMILES string to begin.", None
        if len(text) > MAX_LENGTH:
            return f"Too long - {len(text)} characters, maximum {MAX_LENGTH}.", None
        if any(c.isspace() for c in text):
            return "SMILES strings cannot contain spaces.", None
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return "Not a valid SMILES string - check for unclosed rings or brackets.", None
        if mol.GetNumHeavyAtoms() == 0:
            return "That parses, but contains no heavy atoms.", None
        return (f"{rdMolDescriptors.CalcMolFormula(mol)} · "
                f"{mol.GetNumHeavyAtoms()} heavy atoms"), mol
    except Exception:
        return "Could not read that input.", None


def structure_svg(mol, width=420, height=150):
    """2-D depiction of the parsed molecule, so the user can see what was understood."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.bondLineWidth = 2
    options.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().split("?>", 1)[-1]        # drop the XML declaration
    return f"<div class='structure'>{svg}</div>"


def gauge(probability, blocker):
    """Where this probability sits relative to the decision threshold."""
    colour = BLOCK if blocker else SAFE
    return f"""
<div class="gauge">
  <div class="gauge-track">
    <div class="gauge-fill" style="width:{probability * 100:.1f}%;background:{colour}"></div>
    <div class="gauge-mark" style="left:{THRESHOLD * 100:.1f}%"></div>
  </div>
  <div class="gauge-labels">
    <span>0.0 &middot; not a blocker</span>
    <span class="gauge-thr">threshold {THRESHOLD}</span>
    <span>blocker &middot; 1.0</span>
  </div>
</div>"""


def on_type(text):
    message, mol = validate(text)
    ok = mol is not None
    css = "hint ok" if ok else ("hint" if not (text or "").strip() else "hint bad")
    return (gr.update(value=f"<div class='{css}'>{message}</div>"),
            gr.update(interactive=ok),
            gr.update(value=structure_svg(mol) if ok else "", visible=ok))


def predict(text):
    message, mol = validate(text)
    if mol is None:
        return gr.update(visible=True, value=f"<div class='card'><div class='hint bad'>{message}</div></div>")

    probability = float(np.asarray(
        SESSION.run(None, {INPUT_NAME: featurize([text.strip()])})[-1], float)[0, 1])
    blocker = probability >= THRESHOLD
    colour = BLOCK if blocker else SAFE
    margin = abs(probability - THRESHOLD)
    confidence = "high" if margin > 0.25 else "moderate" if margin > 0.10 else "borderline"

    return gr.update(visible=True, value=f"""
<div class="card">
  <div class="verdict" style="color:{colour}">
    {"Likely hERG blocker" if blocker else "Likely not a hERG blocker"}
  </div>
  <div class="figures">
    <div><span class="k">Probability</span><span class="v">{probability:.3f}</span></div>
    <div><span class="k">Predicted class</span><span class="v">{int(blocker)}</span></div>
    <div><span class="k">Confidence</span><span class="v">{confidence}</span></div>
  </div>
  {gauge(probability, blocker)}
  <p class="explain">
    The model estimates a <b>{probability:.1%}</b> chance that this molecule blocks the
    hERG potassium channel. Anything at or above {THRESHOLD} is flagged as a blocker.
    hERG blockade can cause cardiac arrhythmia, so a positive result means the compound
    warrants dedicated cardiac safety testing before it goes further.
  </p>
</div>""")


EXAMPLES = [["CN(C)CCOC(c1ccccc1)c1ccccc1"],
            ["CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1"],
            ["CC(=O)Oc1ccccc1C(=O)O"],
            ["CCO"]]
EXAMPLE_LABELS = ["Diphenhydramine", "Terfenadine", "Aspirin", "Ethanol"]

CSS = """
.gradio-container {max-width: 760px !important;}
footer {display: none !important;}
#title {font-size: 1.9rem; font-weight: 650; letter-spacing: -0.02em; margin-bottom: 4px;}
#subtitle {opacity: 0.68; font-size: 0.95rem; line-height: 1.5; margin-bottom: 4px;}
.hint {font-size: 0.88rem; min-height: 1.3em; opacity: 0.6;}
.hint.ok {color: #1f7a4d; opacity: 1; font-variant-numeric: tabular-nums;}
.hint.bad {color: #c5283d; opacity: 1;}
.structure {background: #ffffff; border-radius: 10px; padding: 4px 10px; margin: 2px 0;}
.structure svg {display: block; width: 100%; height: auto;}
.card {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 22px 24px;}
.verdict {font-size: 1.4rem; font-weight: 650; letter-spacing: -0.01em; margin-bottom: 18px;}
.figures {display: flex; gap: 34px; flex-wrap: wrap; margin-bottom: 22px;}
.figures .k {display: block; font-size: 0.72rem; text-transform: uppercase;
             letter-spacing: 0.06em; opacity: 0.55; margin-bottom: 3px;}
.figures .v {display: block; font-size: 1.25rem; font-weight: 600;
             font-variant-numeric: tabular-nums;}
.gauge-track {position: relative; height: 8px; border-radius: 99px;
              background: var(--background-fill-secondary); overflow: visible;}
.gauge-fill {height: 100%; border-radius: 99px;}
.gauge-mark {position: absolute; top: -4px; width: 2px; height: 16px;
             background: var(--body-text-color); opacity: 0.55; border-radius: 1px;}
.gauge-labels {display: flex; justify-content: space-between; margin-top: 7px;
               font-size: 0.72rem; opacity: 0.5;}
.gauge-thr {opacity: 0.8;}
.explain {font-size: 0.89rem; line-height: 1.6; opacity: 0.78; margin: 20px 0 0 0;
          padding-top: 16px; border-top: 1px solid var(--border-color-primary);}
#footnote {font-size: 0.78rem; opacity: 0.5; text-align: center; margin-top: 6px;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="hERG Blockade Predictor") as demo:
    gr.HTML("<div id='title'>hERG Blockade Predictor</div>"
            "<div id='subtitle'>Estimates whether a molecule blocks the hERG potassium "
            "channel - a common cause of drug-induced cardiac arrhythmia, and a standard "
            "safety check in early drug discovery.</div>")

    smiles_box = gr.Textbox(label="SMILES string", show_label=False, autofocus=True,
                            max_lines=1, placeholder="Paste a SMILES string, e.g. CCO")
    hint = gr.HTML("<div class='hint'>Enter a SMILES string to begin.</div>")
    structure = gr.HTML(visible=False)
    submit = gr.Button("Predict", variant="primary", interactive=False)
    result = gr.HTML(visible=False)

    gr.Examples(examples=EXAMPLES, inputs=smiles_box, example_labels=EXAMPLE_LABELS,
                label="Or try a known compound")

    gr.HTML(f"<div id='footnote'>Extremely randomised trees on {N_FEATURES} physicochemical "
            f"and structural descriptors &middot; decision threshold {THRESHOLD} &middot; "
            f"coursework use only, not a substitute for experimental cardiac safety "
            f"assessment.</div>")

    smiles_box.change(on_type, smiles_box, [hint, submit, structure], show_progress="hidden")
    smiles_box.submit(predict, smiles_box, result)
    submit.click(predict, smiles_box, result)

if __name__ == "__main__":
    # Hosts such as Render supply the port in $PORT; locally this falls back to 7860.
    import os
    # SHARE=1 opens a public tunnel (Gradio relays to this process) when we are not
    # running behind a host that already gives us a public address.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                share=os.environ.get("SHARE") == "1")
