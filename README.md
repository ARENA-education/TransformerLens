> [!NOTE]
> This section (down to the `---` divider below) was written by Claude (Anthropic), an AI
> assistant, while investigating and fixing a dependency conflict for the ARENA course on behalf
> of David Quarel. It is not authored by David.

# ARENA fork: `v2.18.0-numpy-fix` branch

This branch is [ARENA](https://arena.education)'s patched fork of TransformerLens, based on the
last pre-3.0 release, [`v2.18.0`](https://github.com/TransformerLensOrg/TransformerLens/releases/tag/v2.18.0).
ARENA's course material pins `transformer_lens>=2.16.1,<3.0.0` (upgrading to the 3.x line is a
breaking change for the exercises that isn't feasible right now), but the real `v2.18.0` release on
PyPI can't be installed alongside `jax[cuda12]==0.10.1`, which ARENA's course also requires for a
separate (RL/MuJoCo) chapter. This branch carries the minimal set of `pyproject.toml` patches needed
to resolve that conflict, with no other source changes.

## Patches on top of `v2.18.0`

1. **Relax the `numpy<2` pin** ([`31f154d`](https://github.com/ARENA-education/TransformerLens/commit/31f154d0fa74de94cb742997a14bddd98c729527)).
   `v2.18.0`'s `pyproject.toml` caps numpy at `<2.0` for Python <3.12, which is unsatisfiable
   alongside `jax[cuda12]==0.10.1` (which requires `numpy>=2.0`). This cap was added upstream in
   [`a36b57d`](https://github.com/TransformerLensOrg/TransformerLens/commit/a36b57dee1cdee7270b1e0d5ef259127b69b4dbf)
   ("updated numpy dependency", [#943](https://github.com/TransformerLensOrg/TransformerLens/pull/943))
   as a pure dependency-metadata change (no source files touched), and was itself removed again
   upstream in [`60946c4`](https://github.com/TransformerLensOrg/TransformerLens/commit/60946c47ec9a0277dc105c83cc830afabf527104)
   ("removed numpy ceiling", [#994](https://github.com/TransformerLensOrg/TransformerLens/pull/994))
   — but only as part of the 3.0 release, never backported to 2.x. Nothing in `v2.18.0`'s source
   actually depends on numpy<2 behavior (verified: a real `HookedTransformer` forward pass,
   `run_with_cache`, and `utils.to_numpy` all work unmodified under numpy 2.4.6), so this patch just
   cherry-picks the dependency relaxation onto the last 2.x release.

2. **Relax the `pandas<2.1` pin** ([`7990062`](https://github.com/ARENA-education/TransformerLens/commit/7990062c2bf52d39140577eb32b1a38e210941a4)).
   Installing numpy>=2 alone isn't sufficient: `v2.18.0`'s `pandas<2.1` cap (for Python <3.12) forces
   a pandas build compiled against numpy 1.x's C ABI, which fails on import against numpy 2 with
   `ValueError: numpy.dtype size changed, may indicate binary incompatibility`. This cap was added
   upstream in [`833fb95`](https://github.com/TransformerLensOrg/TransformerLens/commit/833fb958e63c78b09f2a79893aa03c5b0ba78c58)
   ("Isolate demo dependencies and pin orjson for CVE-2025-67221 mitigation",
   [#1173](https://github.com/TransformerLensOrg/TransformerLens/pull/1173)) — an unrelated
   security-patch commit that bundled in this pandas cap with no discussion or source changes.
   This patch reverts to the unconstrained `pandas>=1.1.5` that was already in place at `v2.16.0`/
   `v2.15.4` (before that commit), matching how the numpy pin above was relaxed. Verified working:
   `pandas>=2.1` resolves and imports cleanly alongside numpy 2.

3. **Set `version` to `2.18.0.post1`** ([`afa5b9a`](https://github.com/ARENA-education/TransformerLens/commit/afa5b9abc35a29083a045f353761174e69e81a8c)).
   Upstream's `pyproject.toml` hardcodes `version="0.0.0"`, normally bumped by their release
   pipeline before publishing to PyPI — a step that never runs on this git-installed fork. Left as
   `0.0.0`, any dependent with a version-range constraint fails: e.g. ARENA also installs
   `sae-vis==0.3.7`, which requires `transformer-lens>=2.0.0,<3.0.0`, and `0.0.0` doesn't satisfy
   that range. Setting the version explicitly (as a `.post1` of the real `2.18.0` base) fixes
   dependency resolution for anything checking the installed version.

## Why not just upgrade to TransformerLens 3.0?

Both pins above were lifted upstream, but only as part of the 3.0 release, which also brought
[real breaking changes](https://github.com/TransformerLensOrg/TransformerLens/compare/v2.18.0...v3.0.0)
unrelated to numpy/pandas. ARENA's exercises are pinned to the 2.x API and can't absorb that
upgrade right now, so this branch exists to backport just the dependency fix without the breaking
API changes.

---

# TransformerLens

<!-- Status Icons -->
[![Pypi](https://img.shields.io/pypi/v/transformer-lens?color=blue)](https://pypi.org/project/transformer-lens/)
![Pypi Total Downloads](https://img.shields.io/pepy/dt/transformer_lens?color=blue) ![PyPI -
License](https://img.shields.io/pypi/l/transformer_lens?color=blue) [![Release
CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/release.yml/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/release.yml)
[![Tests
CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/checks.yml/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/checks.yml)
[![Docs
CD](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/TransformerLensOrg/TransformerLens/actions/workflows/pages/pages-build-deployment)

A Library for Mechanistic Interpretability of Generative Language Models. Maintained by [Bryce Meyer](https://github.com/bryce13950) and created by [Neel Nanda](https://neelnanda.io/about)

[![Read the Docs
Here](https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white&link=https://TransformerLensOrg.github.io/TransformerLens/)](https://TransformerLensOrg.github.io/TransformerLens/)

This is a library for doing [mechanistic
interpretability](https://distill.pub/2020/circuits/zoom-in/) of GPT-2 Style language models. The
goal of mechanistic interpretability is to take a trained model and reverse engineer the algorithms
the model learned during training from its weights.

TransformerLens lets you load in 50+ different open source language models, and exposes the internal
activations of the model to you. You can cache any internal activation in the model, and add in
functions to edit, remove or replace these activations as the model runs.

## Quick Start

### Install

```shell
pip install transformer_lens
```

### Use

```python
import transformer_lens

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")
```

## Key Tutorials

* [Introduction to the Library and Mech
  Interp](https://arena-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp)
* [Demo of Main TransformerLens Features](https://neelnanda.io/transformer-lens-demo)

## Gallery

Research done involving TransformerLens:

<!-- If you change this also change docs/source/content/gallery.md -->
* [Progress Measures for Grokking via Mechanistic
  Interpretability](https://arxiv.org/abs/2301.05217) (ICLR Spotlight, 2023) by Neel Nanda, Lawrence
  Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt
* [Finding Neurons in a Haystack: Case Studies with Sparse
  Probing](https://arxiv.org/abs/2305.01610) by Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine
  Harvey, Dmitrii Troitskii, Dimitris Bertsimas
* [Towards Automated Circuit Discovery for Mechanistic
  Interpretability](https://arxiv.org/abs/2304.14997) by Arthur Conmy, Augustine N. Mavor-Parker,
  Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso
* [Actually, Othello-GPT Has A Linear Emergent World Representation](https://neelnanda.io/othello)
  by Neel Nanda
* [A circuit for Python docstrings in a 4-layer attention-only
  transformer](https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn/a-circuit-for-python-docstrings-in-a-4-layer-attention-only)
  by Stefan Heimersheim and Jett Janiak
* [A Toy Model of Universality](https://arxiv.org/abs/2302.03025) (ICML, 2023) by Bilal Chughtai,
  Lawrence Chan, Neel Nanda
* [N2G: A Scalable Approach for Quantifying Interpretable Neuron Representations in Large Language
  Models](https://openreview.net/forum?id=ZB6bK6MTYq) (2023, ICLR Workshop RTML) by Alex Foote, Neel
  Nanda, Esben Kran, Ioannis Konstas, Fazl Barez
* [Eliciting Latent Predictions from Transformers with the Tuned
  Lens](https://arxiv.org/abs/2303.08112) by Nora Belrose, Zach Furman, Logan Smith, Danny Halawi,
  Igor Ostrovsky, Lev McKinney, Stella Biderman, Jacob Steinhardt

User contributed examples of the library being used in action:

* [Induction Heads Phase Change
  Replication](https://colab.research.google.com/github/ckkissane/induction-heads-transformer-lens/blob/main/Induction_Heads_Phase_Change.ipynb):
  A partial replication of [In-Context Learning and Induction
  Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
  from Connor Kissane
* [Decision Transformer
  Interpretability](https://github.com/jbloomAus/DecisionTransformerInterpretability): A set of
  scripts for training decision transformers which uses transformer lens to view intermediate
  activations, perform attribution and ablations. A write up of the initial work can be found
  [here](https://www.lesswrong.com/posts/bBuBDJBYHt39Q5zZy/decision-transformer-interpretability).

Check out [our demos folder](https://github.com/TransformerLensOrg/TransformerLens/tree/main/demos) for
more examples of TransformerLens in practice

## Getting Started in Mechanistic Interpretability

Mechanistic interpretability is a very young and small field, and there are a _lot_ of open
problems. This means there's both a lot of low-hanging fruit, and that the bar for entry is low - if
you would like to help, please try working on one! The standard answer to "why has no one done this
yet" is just that there aren't enough people! Key resources:

* [A Guide to Getting Started in Mechanistic Interpretability](https://neelnanda.io/getting-started)
* [ARENA Mechanistic Interpretability Tutorials](https://arena3-chapter1-transformer-interp.streamlit.app/) from
  Callum McDougall. A comprehensive practical introduction to mech interp, written in
  TransformerLens - full of snippets to copy and they come with exercises and solutions! Notable
  tutorials:
  * [Coding GPT-2 from
    scratch](https://arena3-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch), with
    accompanying video tutorial from me ([1](https://neelnanda.io/transformer-tutorial)
    [2](https://neelnanda.io/transformer-tutorial-2)) - a good introduction to transformers
  * [Introduction to Mech Interp and
    TransformerLens](https://arena3-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp): An
    introduction to TransformerLens and mech interp via studying induction heads. Covers the
    foundational concepts of the library
  * [Indirect Object
    Identification](https://arena3-chapter1-transformer-interp.streamlit.app/[1.3]_Indirect_Object_Identification):
    a replication of interpretability in the wild, that covers standard techniques in mech interp
    such as [direct logit
    attribution](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=disz2gTx-jooAcR0a5r8e7LZ),
    [activation patching and path
    patching](https://www.lesswrong.com/posts/xh85KbTFhbCz7taD4/how-to-think-about-activation-patching)
* [Mech Interp Paper Reading List](https://neelnanda.io/paper-list)
* [200 Concrete Open Problems in Mechanistic
  Interpretability](https://neelnanda.io/concrete-open-problems)
* [A Comprehensive Mechanistic Interpretability Explainer](https://neelnanda.io/glossary): To look
  up all the jargon and unfamiliar terms you're going to come across!
* [Neel Nanda's Youtube channel](https://www.youtube.com/channel/UCBMJ0D-omcRay8dh4QT0doQ): A range
  of mech interp video content, including [paper
  walkthroughs](https://www.youtube.com/watch?v=KV5gbOmHbjU&list=PL7m7hLIqA0hpsJYYhlt1WbHHgdfRLM2eY&index=1),
  and [walkthroughs of doing
  research](https://www.youtube.com/watch?v=yo4QvDn-vsU&list=PL7m7hLIqA0hr4dVOgjNwP2zjQGVHKeB7T)

## Support & Community

[![Contributing
Guide](https://img.shields.io/badge/-Contributing%20Guide-blue?style=for-the-badge&logo=GitHub&logoColor=white)](https://TransformerLensOrg.github.io/TransformerLens/content/contributing.html)

If you have issues, questions, feature requests or bug reports, please search the issues to check if
it's already been answered, and if not please raise an issue!

You're also welcome to join the open source mech interp community on
[Slack](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-2n26nfoh1-TzMHrzyW6HiOsmCESxXtyw).
Please use issues for concrete discussions about the package, and Slack for higher bandwidth
discussions about eg supporting important new use cases, or if you want to make substantial
contributions to the library and want a maintainer's opinion. We'd also love for you to come and
share your projects on the Slack!

| :exclamation:  HookedSAETransformer Removed   |
|-----------------------------------------------|

Hooked SAE has been removed from TransformerLens in version 2.0. The functionality is being moved to
[SAELens](http://github.com/jbloomAus/SAELens). For more information on this release, please see the
accompanying
[announcement](https://transformerlensorg.github.io/TransformerLens/content/news/release-2.0.html)
for details on what's new, and the future of TransformerLens.

## Credits

This library was created by **[Neel Nanda](https://neelnanda.io)** and is maintained by **[Bryce Meyer](https://github.com/bryce13950)**.

The core features of TransformerLens were heavily inspired by the interface to [Anthropic's
excellent Garcon tool](https://transformer-circuits.pub/2021/garcon/index.html). Credit to Nelson
Elhage and Chris Olah for building Garcon and showing the value of good infrastructure for enabling
exploratory research!

### Creator's Note (Neel Nanda)

I (Neel Nanda) used to work for the [Anthropic interpretability team](transformer-circuits.pub), and
I wrote this library because after I left and tried doing independent research, I got extremely
frustrated by the state of open source tooling. There's a lot of excellent infrastructure like
HuggingFace and DeepSpeed to _use_ or _train_ models, but very little to dig into their internals
and reverse engineer how they work. **This library tries to solve that**, and to make it easy to get
into the field even if you don't work at an industry org with real infrastructure! One of the great
things about mechanistic interpretability is that you don't need large models or tons of compute.
There are lots of important open problems that can be solved with a small model in a Colab notebook!

### Citation

Please cite this library as:

```BibTeX
@misc{nanda2022transformerlens,
    title = {TransformerLens},
    author = {Neel Nanda and Joseph Bloom},
    year = {2022},
    howpublished = {\url{https://github.com/TransformerLensOrg/TransformerLens}},
}
```
